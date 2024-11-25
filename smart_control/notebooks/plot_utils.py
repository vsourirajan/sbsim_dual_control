import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def format_plot(ax1, xlabel: str, start_time: int, end_time: int, time_zone: str):
    """Formats a plot with common attributes."""
    ax1.set_facecolor('black')
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_major_formatter(
      mdates.DateFormatter('%a %m/%d %H:%M', tz=pytz.timezone(time_zone))
    )
    ax1.grid(color='gray', linestyle='-', linewidth=1.0)
    ax1.set_ylabel(xlabel, color='blue', fontsize=12)
    ax1.set_xlim(left=start_time, right=end_time)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(prop={'size': 10})
    
def plot_occupancy_timeline(ax1, reward_timeseries: pd.DataFrame, time_zone: str):
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]
    ax1.plot(
      local_times,
      reward_timeseries['occupancy'],
      color='cyan',
      marker=None,
      alpha=1,
      lw=2,
      linestyle='-',
      label='Num Occupants',
    )
    format_plot(ax1, 'Occupancy', reward_timeseries.index.min(), reward_timeseries.index.max(), time_zone,)
    

def plot_energy_cost_timeline(ax1, reward_timeseries: pd.DataFrame, time_zone: str, cumulative: bool = False):
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]
    if cumulative:
        feature_timeseries_cost = reward_timeseries['electricity_energy_cost'].cumsum()
    else:
        feature_timeseries_cost = reward_timeseries['electricity_energy_cost']
    ax1.plot(
      local_times,
      feature_timeseries_cost,
      color='magenta',
      marker=None,
      alpha=1,
      lw=2,
      linestyle='-',
      label='Electricity',
    )

    format_plot(
      ax1,
      'Energy Cost [$]',
      reward_timeseries.index.min(),
      reward_timeseries.index.max(),
      time_zone,
    )
    
    
def plot_reward_timeline(ax1, reward_timeseries, time_zone):
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    ax1.plot(
      local_times,
      reward_timeseries['cumulative_reward'],
      color='royalblue',
      marker=None,
      alpha=1,
      lw=6,
      linestyle='-',
      label='reward',
    )
    format_plot(
      ax1,
      'Agent Reward',
      reward_timeseries.index.min(),
      reward_timeseries.index.max(),
      time_zone,
    )
    

def plot_energy_timeline(ax1, energy_timeseries, time_zone, cumulative=False):
    def _to_kwh(energy_rate: float, step_interval: pd.Timedelta = pd.Timedelta(5, unit='minute'),) -> float:
        kw_power = energy_rate / 1000.0
        hwh_power = kw_power * step_interval / pd.Timedelta(1, unit='hour')
        return hwh_power.cumsum()

    timeseries = energy_timeseries[
      energy_timeseries['device_type'] == 'air_handler'
    ]

    if cumulative:
        feature_timeseries_ac = _to_kwh(timeseries['air_handler_air_conditioner_energy_rate'])
        feature_timeseries_blower = _to_kwh(timeseries['air_handler_blower_electrical_energy_rate'])
    else:
        feature_timeseries_ac = (timeseries['air_handler_air_conditioner_energy_rate'] / 1000.0)
        feature_timeseries_blower = (timeseries['air_handler_blower_electrical_energy_rate'] / 1000.0)

    ax1.plot(
      timeseries['start_time'],
      feature_timeseries_ac,
      color='magenta',
      marker=None,
      alpha=1,
      lw=4,
      linestyle='-',
      label='AHU Electricity',
    )
    ax1.plot(
      timeseries['start_time'],
      feature_timeseries_blower,
      color='magenta',
      marker=None,
      alpha=1,
      lw=4,
      linestyle='--',
      label='FAN Electricity',
    )

    timeseries = energy_timeseries[energy_timeseries['device_type'] == 'boiler']
    if cumulative:
        feature_timeseries_gas = _to_kwh(
            timeseries['boiler_natural_gas_heating_energy_rate']
        )
        feature_timeseries_pump = _to_kwh(
            timeseries['boiler_pump_electrical_energy_rate']
        )
    else:
        feature_timeseries_gas = (
            timeseries['boiler_natural_gas_heating_energy_rate'] / 1000.0
        )
        feature_timeseries_pump = (
            timeseries['boiler_pump_electrical_energy_rate'] / 1000.0
        )

    ax1.plot(
      timeseries['start_time'],
      feature_timeseries_gas,
      color='lime',
      marker=None,
      alpha=1,
      lw=4,
      linestyle='-',
      label='BLR Gas',
    )
    ax1.plot(
      timeseries['start_time'],
      feature_timeseries_pump,
      color='lime',
      marker=None,
      alpha=1,
      lw=4,
      linestyle='--',
      label='Pump Electricity',
    )

    if cumulative:
        label = 'HVAC Energy Consumption [kWh]'
    else:
        label = 'HVAC Power Consumption [kW]'

    format_plot(ax1, label, timeseries['start_time'].min(), timeseries['end_time'].max(),time_zone,)


def plot_carbon_timeline(ax1, reward_timeseries, time_zone, cumulative=False):
    """Plots carbon-emission timeline."""
    if cumulative:
        feature_timeseries_carbon = reward_timeseries['carbon_emitted'].cumsum()
    else:
        feature_timeseries_carbon = reward_timeseries['carbon_emitted']
    ax1.plot(
      reward_timeseries.index,
      feature_timeseries_carbon,
      color='white',
      marker=None,
      alpha=1,
      lw=4,
      linestyle='-',
      label='Carbon',
    )
    format_plot(ax1, 'Carbon emission [kg]', reward_timeseries.index.min(), reward_timeseries.index.max(), time_zone,)


    
def plot_action_timeline(ax1, action_timeseries, action_tuple, time_zone):
    """Plots action timeline."""
    single_action_timeseries = action_timeseries[
      (action_timeseries['device_id'] == action_tuple[0])
      & (action_timeseries['setpoint_name'] == action_tuple[1])
    ]
    single_action_timeseries = single_action_timeseries.sort_values(by='timestamp')

    if action_tuple[1] in ['supply_water_setpoint','supply_air_heating_temperature_setpoint',]:
        single_action_timeseries['setpoint_value'] = (single_action_timeseries['setpoint_value'] - KELVIN_TO_CELSIUS)

    ax1.plot(
      single_action_timeseries['timestamp'],
      single_action_timeseries['setpoint_value'],
      color='lime',
      marker=None,
      alpha=1,
      lw=4,
      linestyle='-',
      label=action_tuple[1],
    )
    title = '%s %s' % (action_tuple[0], action_tuple[1])
    format_plot(
      ax1,
      'Action',
      single_action_timeseries['timestamp'].min(),
      single_action_timeseries['timestamp'].max(),
      time_zone,
    )




def plot_temperature_timeline(ax1, zone_timeseries, outside_air_temperature_timeseries, time_zone):
    zone_temps = pd.pivot_table(
      zone_timeseries,
      index=zone_timeseries['start_time'],
      columns='zone',
      values='zone_air_temperature',
    ).sort_index()
    zone_temps.quantile(q=0.25, axis=1)
    zone_temp_stats = pd.DataFrame({
      'min_temp': zone_temps.min(axis=1),
      'q25_temp': zone_temps.quantile(q=0.25, axis=1),
      'median_temp': zone_temps.median(axis=1),
      'q75_temp': zone_temps.quantile(q=0.75, axis=1),
      'max_temp': zone_temps.max(axis=1),
    })

    zone_heating_setpoints = (
      pd.pivot_table(
          zone_timeseries,
          index=zone_timeseries['start_time'],
          columns='zone',
          values='heating_setpoint_temperature',
      )
      .sort_index()
      .min(axis=1)
    )
    zone_cooling_setpoints = (
      pd.pivot_table(
          zone_timeseries,
          index=zone_timeseries['start_time'],
          columns='zone',
          values='cooling_setpoint_temperature',
      )
      .sort_index()
      .max(axis=1)
    )

    ax1.plot(
      zone_cooling_setpoints.index,
      zone_cooling_setpoints - KELVIN_TO_CELSIUS,
      color='yellow',
      lw=1,
    )
    ax1.plot(
      zone_cooling_setpoints.index,
      zone_heating_setpoints - KELVIN_TO_CELSIUS,
      color='yellow',
      lw=1,
    )

    ax1.fill_between(
      zone_temp_stats.index,
      zone_temp_stats['min_temp'] - KELVIN_TO_CELSIUS,
      zone_temp_stats['max_temp'] - KELVIN_TO_CELSIUS,
      facecolor='green',
      alpha=0.8,
    )
    ax1.fill_between(
      zone_temp_stats.index,
      zone_temp_stats['q25_temp'] - KELVIN_TO_CELSIUS,
      zone_temp_stats['q75_temp'] - KELVIN_TO_CELSIUS,
      facecolor='green',
      alpha=0.8,
    )
    ax1.plot(
      zone_temp_stats.index,
      zone_temp_stats['median_temp'] - KELVIN_TO_CELSIUS,
      color='white',
      lw=3,
      alpha=1.0,
    )
    ax1.plot(
      outside_air_temperature_timeseries.index,
      outside_air_temperature_timeseries - KELVIN_TO_CELSIUS,
      color='magenta',
      lw=3,
      alpha=1.0,
    )
    format_plot(
      ax1,
      'Temperature [C]',
      zone_temp_stats.index.min(),
      zone_temp_stats.index.max(),
      time_zone,
    )
    
    

def plot_timeseries_charts(reader, time_zone):
    """Plots timeseries charts."""
    observation_responses = reader.read_observation_responses(
      pd.Timestamp.min, pd.Timestamp.max
    )
    action_responses = reader.read_action_responses(
      pd.Timestamp.min, pd.Timestamp.max
    )
    reward_infos = reader.read_reward_infos(pd.Timestamp.min, pd.Timestamp.max)
    reward_responses = reader.read_reward_responses(
      pd.Timestamp.min, pd.Timestamp.max
    )

    if len(reward_infos) == 0 or len(reward_responses) == 0:
        return

    action_timeseries = get_action_timeseries(action_responses)
    action_tuples = list(
      set([
          (row['device_id'], row['setpoint_name'])
          for _, row in action_timeseries.iterrows()
      ])
    )

    reward_timeseries = get_reward_timeseries(
      reward_infos, reward_responses, time_zone
    ).sort_index()
    outside_air_temperature_timeseries = get_outside_air_temperature_timeseries(
      observation_responses, time_zone
    )
    zone_timeseries = get_zone_timeseries(reward_infos, time_zone)
    fig, axes = plt.subplots(
      nrows=6 + len(action_tuples),
      ncols=1,
      gridspec_kw={
          'height_ratios': [1, 1, 1, 1, 1, 1] + [1] * len(action_tuples)
      },
      squeeze=True,
    )
    fig.set_size_inches(24, 25)

    energy_timeseries = get_energy_timeseries(reward_infos, time_zone)
    plot_reward_timeline(axes[0], reward_timeseries, time_zone)
    plot_energy_timeline(axes[1], energy_timeseries, time_zone, cumulative=True)
    plot_energy_cost_timeline(
      axes[2], reward_timeseries, time_zone, cumulative=True
    )
    plot_carbon_timeline(axes[3], reward_timeseries, time_zone, cumulative=True)
    plot_occupancy_timeline(axes[4], reward_timeseries, time_zone)
    plot_temperature_timeline(
      axes[5], zone_timeseries, outside_air_temperature_timeseries, time_zone
    )

    for i, action_tuple in enumerate(action_tuples):
        plot_action_timeline(
            axes[6 + i], action_timeseries, action_tuple, time_zone
        )

    plt.show()

