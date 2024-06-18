from __future__ import annotations

import dataclasses
from collections import Counter
from csv import DictReader, DictWriter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from glob import glob
from os.path import exists as _exists
from os.path import join as _join
from pathlib import Path
from string import Template

import pandas as pd
from dateutil.parser import parse as parse_date
from tabulate import tabulate


def pretty_table(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

def isint(x):
    try:
        return float(int(x)) == float(x)
    except:
        return False


def int_try_parse(x):
    try:
        return int(x)
    except:
        return None


def float_try_parse(x):
    try:
        return float(x)
    except:
        return None


def parse_bool(x):
    return str(x).lower().startswith('true')


def get_time_code(fn, index=-1):
    return fn.split('/')[index][1:]


def calculate_direction(action, last_action):
    if action.value < last_action.value:
        return False
    else:
        return True


def code_procedures(procedures, actions_df):
    df = pd.DataFrame(procedures)
    proc_dict_list = []
    for p, proc in df.iterrows():
        d = {**proc,
             'component_id': 'procedure',
             'bay': actions_df['bay'].iloc[0],
             'pid': actions_df['pid'].iloc[0],
             'block': actions_df['block'].iloc[0]}
        d['step_id'] = d['step_number']
        if d['auto_executed']:
            if not d['mark_value']:
                d['action'] = 'open'
                d['step_id'] = 1
        else:
            if not d['mark_value']:
                d['action'] = 'unchecked'
            else:
                d['action'] = 'checked'
        d['procedure_id'] = d['procedure_id'].split(':')[1]
        proc_dict_list.append(d)
    return pd.DataFrame(proc_dict_list)


def map_actions(ivs, actions, procedures, unit, trial_id, block):
    """
    map_actions codes a list of actions (from trial.actions) based ontimestamps cross-referenced against procedures
    (from trial.procedure_exeuctions). It uses a dictionary to collect the list of actions and converts that to a
    dataframe (df) so that the procedures can be integrated more easily with a sort function.
    This calls another class (SliderActions) that handles removing duplicates due to Tim's overly ambitious 100
    mhz recording of actions that blows up the slider event list size
    """
    dict_list = []
    last_action_timestep = None
    for i, act in enumerate(actions):
        # why are trials showing up that shouldn't be under this directory???
        if act.trial == trial_id:
            d = {**ivs, **act.asdict(), **unit[act.simstep]}
            last_timestep = None
            last_step_number = ''
            d['procedure_id'] = 'unknown'
            d['step_id'] = 'unknown'
            for p_idx, proc in enumerate(procedures):
                timestep = proc.timestamp
                if last_timestep is not None:
                    if p_idx + 1 == len(procedures):
                        if timestep < act.timestamp:
                            # print('last step of trial action = ', act.component_id, ' value = ', act.value)
                            d['procedure_id'] = proc.procedure_id.split(':')[1]
                            d['remaining_step_time'] = None
                            d['total_step_time'] = None
                            d['elapsed_step_time'] = act.timestamp - timestep
                            if last_action_timestep:
                                d['action_delta_time'] = act.timestamp - last_action_timestep
                            else:
                                d['action_delta_time'] = None
                            d[
                                'step_id'] = proc.step_number  # NOTE this is the current step number since it occurs after it has been marked as complete
                            last_action_timestep = act.timestamp
                            break
                    if last_timestep < act.timestamp < timestep:
                        d['procedure_id'] = proc.procedure_id.split(':')[1]
                        d['remaining_step_time'] = timestep - act.timestamp
                        d['total_step_time'] = timestep - last_timestep
                        d['elapsed_step_time'] = act.timestamp - last_timestep
                        if last_action_timestep:
                            d['action_delta_time'] = act.timestamp - last_action_timestep
                        else:
                            d['action_delta_time'] = None
                        if last_step_number == '':
                            d['step_id'] = 1
                        else:
                            d['step_id'] = proc.step_number
                        last_action_timestep = act.timestamp
                        break  # breaks out of trial.procedure_execution for loop to line 496 d['block] = blk_counter[blk_key]

                last_step_number = proc.step_number
                last_timestep = timestep
        d['block'] = block
        dict_list.append(d)

    df = pd.DataFrame(dict_list)
    timedelta_columns = ['remaining_step_time', 'total_step_time', 'elapsed_step_time', 'action_delta_time',
                         'redirection_duration', 'action_duration']

    # Fix timestamps for easier reading later and copy them to string representation columns
    if not df.empty:
        df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S.%f%z")
        for col in timedelta_columns:
            df['{}_str'.format(col)] = df[col].apply(strfdelta)
    return df


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt='%H:%M:%S.%f'):
    if pd.isnull(tdelta):
        return 'NaT'
    else:
        d = {"H": None}
        hours, rem = divmod(tdelta.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds, microseconds = divmod(tdelta.microseconds, 1000)
        d["H"] = '{:02d}'.format(hours)
        d["M"] = '{:02d}'.format(minutes)
        d["S"] = '{:02d}'.format(seconds)
        d["f"] = '{:03d}'.format(milliseconds)
        t = DeltaTemplate(fmt)
        return t.substitute(**d)


class SimulationStatus(Enum):
    NOT_STARTED = 0
    STARTED = 2
    PAUSED = 3
    STOPPED = 4


class EventCollectionMixin:
    def __iter__(self):
        for simstep, evt in self.events.items():
            yield evt

    def __getitem__(self, simstep):
        if len(self.events) < simstep:
            return {'quality': 'error', 'quality_msg': 'simstep_missing'}
        else:
            return self.events[simstep]


    def __len__(self):
        return len(self.events)


@dataclass
class SimulationEvent:
    time_code: str
    simstep: int
    timestamp: datetime
    status: SimulationStatus

    def __init__(self, **kwargs):
        self.simstep = int(kwargs['simstep'])
        self.timestamp = parse_date(kwargs['timestamp'])
        self.status = SimulationStatus[kwargs['status']]

    def asdict(self):
        return dataclasses.asdict(self)


class Simulation:
    def __init__(self, fn):
        with open(fn) as fp:
            self.events = [SimulationEvent(**row) for row in DictReader(fp)]

    @property
    def qc(self):
        return self.start_time is not None and self.stop_time is not None

    @property
    def last_simstep(self):
        if self.events:
            return self.events[-1].simstep

    @property
    def start_time(self):
        for evt in self.events:
            if evt.status == SimulationStatus.STARTED:
                return evt.timestamp

    @property
    def stop_time(self):
        # iterate in verse and return first stop event
        for evt in self.events[::-1]:
            if evt.status == SimulationStatus.STOPPED:
                return evt.timestamp

    @property
    def elapsed_time(self):
        stop_time = self.stop_time
        start_time = self.start_time
        if start_time and stop_time:
            return stop_time - start_time

    def asdict(self):
        return dict(last_simstep=self.last_simstep,
                    start_time=self.start_time, stop_time=self.stop_time,
                    elapsed_time=self.elapsed_time, qc=self.qc)


@dataclass
class Action:
    time_code: str
    simstep: int
    timestamp: datetime
    reactor_unit: int
    action: str
    component_id: str
    value: str
    redirection_duration: timedelta
    action_duration: timedelta
    redirections: int = 0
    procedure_id: str = ''
    step_id: str = ''

    def __init__(self, **kwargs):
        self.time_code = kwargs['time_code']
        self.simstep = int(kwargs['simstep'])
        self.timestamp = parse_date(kwargs['initiation timestamp'])
        self.reactor_unit = kwargs['reactor unit']
        self.action = kwargs['action']
        self.component_id = kwargs['component id or recipient']
        self.value = kwargs['value']
        self.redirection_duration = timedelta(milliseconds=0)
        self.action_duration = timedelta(milliseconds=0)

    def __eq__(self, other):
        return self.action, self.component_id, self.value == other.action, other.component_id, other.value

    def __ne__(self, other):
        return self.action != other.action or \
               self.component_id != other.component_id or \
               self.value != other.value

    def asdict(self):
        return dataclasses.asdict(self)


class Actions(EventCollectionMixin):
    def __init__(self, fn=None):
        if fn:
            consecutive_variables = ['bypass_demand', 'governor_demand', 'control_demand', 'reactivity_rx_target',
                                     'reactivity_temperature_target', 'fw_cv_a_demand', 'fw_cv_b_demand']
            events = []
            id_column = 'component id or recipient'
            time_column = 'initiation timestamp'
            with open(fn) as fp:
                curr_comp_id = None
                last_comp_id = None
                curr_val = None
                last_value = None
                last_time = None
                duration = None
                last_consecutive_id = None
                consecutive_actions = []

                # length = len(list(DictReader(fp)))
                for r, row in enumerate(DictReader(fp)):
                    comp_id = row[id_column]
                    row_dict = {**row, 'time_code': get_time_code(fn, -2)}
                    if comp_id == last_comp_id:
                        if comp_id in consecutive_variables:

                            consecutive_actions.append(Action(**row_dict))
                            last_consecutive_id = comp_id
                        else:
                            events.append(Action(**row_dict))
                    else:
                        if last_comp_id in consecutive_variables:
                            if last_consecutive_id is not None:
                                # add ending series of slider_actions as one special slider_action
                                if len(consecutive_actions) > 0:
                                    slider_series = SliderActionSeries(consecutive_actions)
                                    actions = slider_series.get_action_results()
                                    for sact in actions:
                                        events.append(sact)
                                    consecutive_actions = []
                                last_consecutive_id = None
                        if comp_id in consecutive_variables:
                            consecutive_actions.append(Action(**row_dict))
                            last_consecutive_id = comp_id
                        else:
                            #add next event that is not part of a series of slider_actions, checks if the next id will be the same
                            events.append(Action(**row_dict))
                    last_comp_id = str(comp_id)
                if last_comp_id in consecutive_variables:
                    if len(consecutive_actions) > 0:
                        slider_series = SliderActionSeries(consecutive_actions)
                        actions = slider_series.get_action_results()
                        for sact in actions:
                            events.append(sact)
            self.events = {evt.simstep: evt for evt in events}


class SliderActionSeries(EventCollectionMixin):
    slider_series = []
    action_results = []
    debounce_action_threshold = timedelta(milliseconds=1000)
    debounce_slider_threshold = timedelta(milliseconds=50)

    def __init__(self, slider_actions):
        assert slider_actions is not None and len(slider_actions) > 0
        last_t = None
        last_value = None
        base_direction = 0
        last_direction = 0

        last_value = None
        last_value = None
        sub_slider_actions = []
        for sact in slider_actions:
            t = sact.timestamp
            if '=' in sact.value:
                incr = float(str(sact.value).split('=')[1])
                if last_value:
                    sact.value = last_value + float(incr)
                else:
                    sact.value = float(incr)
            else:
                sact.value = float(sact.value)

            if last_t is not None:
                if self.debounce_action_threshold > last_t - t:
                    sub_slider_actions.append(sact)
                else:
                    self.slider_series.append(sub_slider_actions)
                    sub_slider_actions = [sact]
            else:
                sub_slider_actions.append(sact)
            last_t = t
            last_value = float(sact.value)
        if len(sub_slider_actions) > 0:
            self.slider_series.append(sub_slider_actions)


        for sub_series_acts in self.slider_series:
            sub_base_act = None
            if len(sub_series_acts) > 1:
                base_direction = float(sub_series_acts[-1].value) - float(sub_series_acts[0].value)
            last_direction = None
            for idx, sact in enumerate(sub_series_acts):
                value = float(sact.value)
                t = sact.timestamp
                if idx == 0:
                    sub_base_act = sact
                    sub_base_act.timestamp = t
                else:
                    duration = t - last_t
                    sub_base_act.action_duration += duration
                    direction = last_value - value
                    if base_direction != 0:
                        if (direction / base_direction) < 0:
                            sub_base_act.redirection_duration += duration
                        if duration > self.debounce_slider_threshold:
                            sub_base_act.redirections += 1
                    last_direction = direction
                last_value = value
                last_t = t
            sub_base_act.value = last_value
            self.action_results.append(sub_base_act)


    def get_action_results(self):
        return self.action_results


@dataclass
class ProcedureEvent:
    time_code: str
    simstep: int
    timestamp: datetime
    auto_executed: bool
    procedure_id: str
    step_number: str
    location_info: str
    mark_type: str
    mark_value: bool

    def __init__(self, **kwargs):
        self.time_code = kwargs['time_code']
        self.simstep = int(kwargs['simstep'])
        self.timestamp = parse_date(kwargs['timestamp'])
        self.auto_executed = parse_bool(kwargs['auto executed'])
        self.procedure_id = kwargs['procedure id']
        self.step_number = kwargs['step number']
        self.location_info = kwargs['location info']
        self.mark_type = kwargs['mark type']
        self.mark_value = parse_bool(kwargs['mark value'])

    def asdict(self):
        return dataclasses.asdict(self)


class Procedures(EventCollectionMixin):
    def __init__(self, fns):
        events = []
        for fn in fns:
            with open(fn) as fp:
                events.extend([ProcedureEvent(**row, time_code=get_time_code(fn)) for row in DictReader(fp)])
        events = sorted(events, key=lambda evt: evt.simstep)
        self.events = {evt.simstep: evt for evt in events}


@dataclass
class Interaction:
    simstep: int
    timestamp: datetime
    touch_x: int | None
    touch_y: int | None
    mouse_x: int | None
    mouse_y: int | None
    duration_ms: float

    def __init__(self, **kwargs):
        self.simstep = int(kwargs['simstep'])
        self.timestamp = parse_date(kwargs['timestamp'])
        self.touch_x = int_try_parse(kwargs['touch x'])
        self.touch_y = int_try_parse(kwargs['touch y'])
        self.mouse_x = int_try_parse(kwargs['mouse x'])
        self.mouse_y = int_try_parse(kwargs['mouse y'])
        self.duration_ms = float_try_parse(kwargs['duration ms'])

    def asdict(self):
        return dataclasses.asdict(self)


class Interactions(EventCollectionMixin):
    def __init__(self, fn):
        with open(fn) as fp:
            self.events = {int(row['simstep']): Interaction(**row) for row in DictReader(fp)}


class Unit(EventCollectionMixin):
    def __init__(self, fn):
        with open(fn) as fp:
            # store as a dictionary with simstep keys this makes it easy to query the plant state
            self.events = {int(row['simstep']): row for row in DictReader(fp)}


class Trial:
    def __init__(self, trial_dir, ivs):
        assert _exists(trial_dir)
        self.time_code = get_time_code(str(trial_dir))
        self._dir = trial_dir
        self.ivs = ivs

    @property
    def action_events(self):
        return Actions(_join(self._dir, 'actions.csv'))

    @property
    def interactions(self):
        return Interactions(_join(self._dir, 'interaction.csv'))

    @property
    def procedure_events(self):
        return Procedures(glob(_join(self._dir, 'procedure_execution_*.csv')))

    @property
    def simulation(self):
        return Simulation(_join(self._dir, 'simulation.csv'))

    @property
    def unit(self):
        return Unit(_join(self._dir, 'unit_1.csv'))

    def asdict(self):
        return {**self.ivs, **self.simulation.asdict()}

    def __str__(self):
        return f'Trial({self._dir})'


class TrialManager:
    def __init__(self, data_dir='../data', ivs_func=None):
        self.data_dir = data_dir
        p = Path(data_dir)
        self.trials = []

        for x in p.glob('**/unit_1.csv'):
            trial_dir = x.parents[0]
            self.trials.append(Trial(trial_dir, ivs_func(trial_dir)))

    def dump_qc_summary_csv(self, fn, filter_func=None):
        if filter_func:
            trials = filter(filter_func, self.trials)
        else:
            trials = self.trials

        with open(fn, 'w', newline='') as fp:
            wtr = DictWriter(fp, fieldnames=trials[0].asdict().keys())
            wtr.writeheader()
            for trial in trials:
                wtr.writerow(trial.asdict())

    def dump_procedure_events(self, fn, filter_func=None):
        if filter_func:
            trials = filter(filter_func, self.trials)
        else:
            trials = self.trials

        blk_counter = Counter()

        with open(fn, 'w', newline='') as fp:
            init = 1
            for trial in trials:
                last_t = None
                last_step_number = None
                unit_1 = trial.unit

                blk_key = (trial.ivs['pid'], trial.ivs['scenario'])
                blk_counter[blk_key] += 1
                print(blk_key, blk_counter[blk_key])

                for proc in trial.procedure_events:
                    proc.procedure_id = proc.procedure_id.split(':')[1]

                    d = {**trial.ivs, **proc.asdict(), **unit_1[proc.simstep]}
                    t = parse_date(d['timestamp'])

                    ###Label Actions by Step####
                    for act in trial.action_events:
                        if last_t is not None:
                            if last_t < act.timestamp < t:
                                act.procedure_id = proc.procedure_id
                                if last_step_number == '':
                                    act.step_id = 0
                                else:
                                    act.step_id = last_step_number

                    if last_t is not None:
                        d['delta_time'] = t - last_t
                    else:
                        d['delta_time'] = None

                    d['block'] = blk_counter[blk_key]

                    if init:
                        wtr = DictWriter(fp, fieldnames=d.keys())
                        wtr.writeheader()
                        init = 0

                    last_t = t
                    last_step_number = proc.step_number
                    wtr.writerow(d)




    ##############################
    #
    # Modified for Actions Compiling by Step (Tom)
    #
    ##############################
    def dump_action_events(self, fn, filter_func=None):
        """
        Codes actions based on procedures and combine thosse into a single output file with each row representing a procedure
        step or an action.
        """
        if filter_func:
            trials = filter(filter_func, self.trials)
        else:
            trials = self.trials

        blk_counter = Counter()

        action_headers = []
        unit_headers = []
        iv_headers = []

        trial_count = len(self.trials)-1

        with open(fn, 'w', newline='') as fp:

            for i, trial in enumerate(trials):
                unit = trial.unit
                blk_key = (trial.ivs['pid'], trial.ivs['scenario'])
                blk_counter[blk_key] += 1
                print('{}%'.format(round(i/trial_count*100, 2)), blk_key, blk_counter[blk_key], trial.time_code)

                coded_actions_df = map_actions(trial.ivs, trial.action_events, trial.procedure_events, trial.unit, trial.time_code, blk_counter)
                coded_procedures_df = code_procedures(trial.procedure_events, coded_actions_df)

                df = None

                if not coded_procedures_df.empty:
                    df = pd.concat([df, coded_procedures_df], ignore_index=True)
                    # headers = list(chain(action_headers, procedure_headers, iv_headers, unit_headers))
                    # df = df[[headers]]
                    df = df.drop(['date', 'reactor_unit', 'location_info', 'mark_type', 'bay'], axis=1)
                    df = df.sort_values(by=['timestamp'])

                if not df.empty:
                    action_procedure_dict = df.to_dict('records')
                    wtr = DictWriter(fp, fieldnames=action_procedure_dict[0].keys())
                    wtr.writeheader()
                    for row in action_procedure_dict:
                        wtr.writerow(row)

    def __iter__(self):
        for trial in self.trials:
            yield trial


def unpack_ivs(path):
    """
    takes a trial directory path and unpacks independent variables. returns ivs as a dict
    trial manager should be agnostic to the variables.

    analyst need to define this. This one is for the directory structure of the summer 2022 intern study data
    """
    parts = path.parts
    date = parts[-1]
    date = date.split('_')
    year, month, day, hour, minute, second, microsecond, _ = [int(v) for v in date if isint(v)]
    date = datetime(year, month, day, hour, minute, second, microsecond)

    scenario = parts[-2]

    try:
        session, cbp_type, pid, bay = parts[-3].split('_')
    except:
        session, cbp_type, pid, bay = parts[-4].split('_')
        scenario = parts[-3]
        bay = parts[-2]

    return dict(date=date, scenario=scenario, session=session,
                cbp_type=cbp_type, pid=pid, bay=bay)


if __name__ == "__main__":
    print('loading TrialManager')
    # data / qc_trial_summary.csv
    data_dir = r'../data'
    mgr = TrialManager(data_dir=data_dir, ivs_func=unpack_ivs)
    # data / qc_trial_summary.csv
    # print('qc dump')
    # mgr.dump_qc_summary_csv(_join(data_dir, 'qc_trial_summary.csv'))

    # fp = open(_join(data_dir, 'performance_dvs.csv'), 'w', newline='')
    # wtr = DictWriter(fp, fieldnames=('date', 'scenario', 'session', '_dir',
    #                                  'cbp_type', 'pid', 'bay', 'repetition', 'k', 'collection_rate',
    #                                  'generated_value_0_usd', 'generated_value_end_usd',
    #                                  'generated_value_usd',
    #                                  'core_temperature_steps_okay',
    #                                  'core_temperature_steps_not_okay',
    #                                  'core_temperature_steps_okay_s',
    #                                  'core_temperature_steps_not_okay_s', 'actions_performed', 'elapsed_time_s'))
    # wtr.writeheader()

    print('scenario analysis...')

    scn = 'startup'

    filter_func = lambda t: t.simulation.qc and t.ivs['scenario'] == scn
    print(f'    dumping action execution')
    mgr.dump_action_events(_join(data_dir, f'{scn}_aggregated_action_execution.csv'), filter_func=filter_func)

    # for scn in ('feedwater', 'startup'):
    #
    #     pid_reps = Counter()
    #     print(f'  {scn}')
    #
    #     filter_func = lambda t: t.simulation.qc and t.ivs['scenario'] == scn
    #
    #     # print(f'    dumping procedure execution')
    #     # mgr.dump_procedure_execution(
    #     #     _join(data_dir, f'{scn}_aggre'
    #     #                     f'gated_proc_execution.csv'),
    #     #     filter_func=filter_func)
    #
    #     print(f'    dumping action execution')
    #     mgr.dump_action_execution(_join(data_dir, f'{scn}_aggregated_action_execution.csv'),
    #                               filter_func=filter_func)

        # iterate over trials and calculate performance measures
        # print(f'    calculating performance measures')
        # for trial in filter(filter_func, mgr):
        #     print(f'    {trial}')
        #     actions = trial.actions
        #     simulation = trial.simulation
        #     pid = trial.ivs['pid']
        #
        #     pid_reps[pid] += 1
        #     trial.ivs['repetition'] = pid_reps[pid]
        #
        #     unit_1 = trial.unit_1
        #     core_temp = Counter()
        #     gen_val_0 = None
        #     for k, step in enumerate(unit_1):
        #         gen_val = float(step['generated_value'].replace('$', ''))
        #         if k == 0:
        #             gen_val_0 = gen_val
        #
        #         core_t = float(step['core_temp'])
        #         core_temp[400 < core_t < 650] += 1
        #
        #     print(gen_val, core_temp)
        #     collection_rate = k / simulation.elapsed_time.seconds
        #     wtr.writerow({**trial.ivs, **dict(
        #         _dir=trial._dir,
        #         k=k, collection_rate=k/simulation.elapsed_time.seconds,
        #         generated_value_0_usd=gen_val_0,
        #         generated_value_end_usd=gen_val,
        #         generated_value_usd=gen_val - gen_val_0,
        #         core_temperature_steps_okay_s=core_temp[True] / collection_rate,
        #         core_temperature_steps_not_okay_s=core_temp[False] / collection_rate,
        #         core_temperature_steps_okay=core_temp[True],
        #         core_temperature_steps_not_okay=core_temp[False],
        #         actions_performed=len(actions),
        #         elapsed_time_s=simulation.elapsed_time.seconds)})

    # fp.close()
