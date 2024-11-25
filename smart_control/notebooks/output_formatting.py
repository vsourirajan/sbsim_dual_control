import pandas as pd
from smart_control.utils import conversion_utils


def get_energy_timeseries(reward_infos, time_zone: str) -> pd.DataFrame:
    """Returns a timeseries of energy rates."""

    start_times = []
    end_times = []

    device_ids = []
    device_types = []
    air_handler_blower_electrical_energy_rates = []
    air_handler_air_conditioner_energy_rates = []
    boiler_natural_gas_heating_energy_rates = []
    boiler_pump_electrical_energy_rates = []

    for reward_info in reward_infos:
        end_timestamp = conversion_utils.proto_to_pandas_timestamp(reward_info.end_timestamp).tz_convert(time_zone)
        start_timestamp = end_timestamp - pd.Timedelta(300, unit='second')

        for air_handler_id in reward_info.air_handler_reward_infos:
            start_times.append(start_timestamp)
            end_times.append(end_timestamp)

            device_ids.append(air_handler_id)
            device_types.append('air_handler')

            air_handler_blower_electrical_energy_rates.append(
              reward_info.air_handler_reward_infos[
                  air_handler_id
              ].blower_electrical_energy_rate
            )
            air_handler_air_conditioner_energy_rates.append(
              reward_info.air_handler_reward_infos[
                  air_handler_id
              ].air_conditioning_electrical_energy_rate
            )
            boiler_natural_gas_heating_energy_rates.append(0)
            boiler_pump_electrical_energy_rates.append(0)

        for boiler_id in reward_info.boiler_reward_infos:
            start_times.append(start_timestamp)
            end_times.append(end_timestamp)

            device_ids.append(boiler_id)
            device_types.append('boiler')

            air_handler_blower_electrical_energy_rates.append(0)
            air_handler_air_conditioner_energy_rates.append(0)

            boiler_natural_gas_heating_energy_rates.append(
              reward_info.boiler_reward_infos[
                  boiler_id
              ].natural_gas_heating_energy_rate
            )
            boiler_pump_electrical_energy_rates.append(
              reward_info.boiler_reward_infos[boiler_id].pump_electrical_energy_rate
            )

    df_map = {
        'start_time': start_times,
        'end_time': end_times,
        'device_id': device_ids,
        'device_type': device_types,
        'air_handler_blower_electrical_energy_rate': (air_handler_blower_electrical_energy_rates),
        'air_handler_air_conditioner_energy_rate': (
          air_handler_air_conditioner_energy_rates
        ),
        'boiler_natural_gas_heating_energy_rate': (
          boiler_natural_gas_heating_energy_rates
        ),
        'boiler_pump_electrical_energy_rate': boiler_pump_electrical_energy_rates,
    }
    df = pd.DataFrame(df_map).sort_values('start_time')
    return df



def get_outside_air_temperature_timeseries(observation_responses, time_zone: str,) -> pd.Series:
    """Returns a timeseries of outside air temperature."""
    temps = []
    for i in range(len(observation_responses)):
        for sor in observation_responses[i].single_observation_responses:
            if (sor.single_observation_request.measurement_name == 'outside_air_temperature_sensor'):                
                temp = [(
                    conversion_utils.proto_to_pandas_timestamp(sor.timestamp).tz_convert(time_zone)
                    - pd.Timedelta(300, unit='second'), sor.continuous_value,
                    )][0]
        temps.append(temp)

    res = list(zip(*temps))
    return pd.Series(res[1], index=res[0]).sort_index()


def get_reward_timeseries(reward_infos, reward_responses, time_zone: str) -> pd.DataFrame:
    """Returns a timeseries of reward values."""
    cols = ['agent_reward_value', 'electricity_energy_cost', 'carbon_emitted', 'occupancy',]
    df = pd.DataFrame(columns=cols)

    for i in range(min(len(reward_responses), len(reward_infos))):
        step_start_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_infos[i].start_timestamp
        ).tz_convert(time_zone)
        step_end_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_infos[i].end_timestamp
        ).tz_convert(time_zone)
        delta_time_sec = (step_end_timestamp - step_start_timestamp).total_seconds()
        occupancy = np.sum([
            reward_infos[i].zone_reward_infos[zone_id].average_occupancy
            for zone_id in reward_infos[i].zone_reward_infos
        ])
        idx = conversion_utils.proto_to_pandas_timestamp(reward_infos[i].start_timestamp).tz_convert(time_zone)
        df.loc[idx] = [reward_responses[i].agent_reward_value,
                reward_responses[i].electricity_energy_cost,
                reward_responses[i].carbon_emitted, occupancy]

    df = df.sort_index()
    df['cumulative_reward'] = df['agent_reward_value'].cumsum()
    logging_info('Cumulative reward: %4.2f' % df.iloc[-1]['cumulative_reward'])
    return df


def get_outside_air_temperature_timeseries(observation_responses, time_zone):
    temps = []
    for i in range(len(observation_responses)):
        temp = [(
                conversion_utils.proto_to_pandas_timestamp(sor.timestamp).tz_convert(time_zone),
                sor.continuous_value,
                )
            for sor in observation_responses[i].single_observation_responses
            if sor.single_observation_request.measurement_name == 'outside_air_temperature_sensor'
        ][0]
        temps.append(temp)

    res = list(zip(*temps))
    return pd.Series(res[1], index=res[0]).sort_index()



def get_zone_timeseries(reward_infos, time_zone):
    """Converts reward infos to a timeseries dataframe."""

    start_times = []
    end_times = []
    zones = []
    heating_setpoints = []
    cooling_setpoints = []
    zone_air_temperatures = []
    air_flow_rate_setpoints = []
    air_flow_rates = []
    average_occupancies = []

    for reward_info in reward_infos:
        start_timestamp = conversion_utils.proto_to_pandas_timestamp(reward_info.end_timestamp).tz_convert(time_zone) - pd.Timedelta(300, unit='second')
        end_timestamp = conversion_utils.proto_to_pandas_timestamp(reward_info.end_timestamp).tz_convert(time_zone)

        for zone_id in reward_info.zone_reward_infos:
            zones.append(zone_id)
            start_times.append(start_timestamp)
            end_times.append(end_timestamp)

            heating_setpoints.append(reward_info.zone_reward_infos[zone_id].heating_setpoint_temperature)
            cooling_setpoints.append(reward_info.zone_reward_infos[zone_id].cooling_setpoint_temperature)
            zone_air_temperatures.append(reward_info.zone_reward_infos[zone_id].zone_air_temperature)
            air_flow_rate_setpoints.append(reward_info.zone_reward_infos[zone_id].air_flow_rate_setpoint)
            air_flow_rates.append(reward_info.zone_reward_infos[zone_id].air_flow_rate)
            average_occupancies.append(reward_info.zone_reward_infos[zone_id].average_occupancy)

    df_map = {
      'start_time': start_times,
      'end_time': end_times,
      'zone': zones,
      'heating_setpoint_temperature': heating_setpoints,
      'cooling_setpoint_temperature': cooling_setpoints,
      'zone_air_temperature': zone_air_temperatures,
      'air_flow_rate_setpoint': air_flow_rate_setpoints,
      'air_flow_rate': air_flow_rates,
      'average_occupancy': average_occupancies,
    }
    return pd.DataFrame(df_map).sort_values('start_time')


def get_action_timeseries(action_responses):
    """Converts action responses to a dataframe."""
    timestamps = []
    device_ids = []
    setpoint_names = []
    setpoint_values = []
    response_types = []
    for action_response in action_responses:
        timestamp = conversion_utils.proto_to_pandas_timestamp(action_response.timestamp)
        for single_action_response in action_response.single_action_responses:
            device_id = single_action_response.request.device_id
            setpoint_name = single_action_response.request.setpoint_name
            setpoint_value = single_action_response.request.continuous_value
            response_type = single_action_response.response_type

            timestamps.append(timestamp)
            device_ids.append(device_id)
            setpoint_names.append(setpoint_name)
            setpoint_values.append(setpoint_value)
            response_types.append(response_type)

    return pd.DataFrame({'timestamp': timestamps, 'device_id': device_ids, 'setpoint_name': setpoint_names, 'setpoint_value': setpoint_values,'response_type': response_types,})

