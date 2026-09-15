#!/usr/bin/env python3
"""Interactive CLI menu for simple one-shot Spot body commands.

Usage:
    python basic_commands.py
"""

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import safe_power_off
from bosdyn.client.robot_command import (RobotCommandClient, blocking_sit,
                                         blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

ROBOT_IP = '192.168.80.3'


def stand(robot, command_client):
    robot.logger.info('Standing up')
    blocking_stand(command_client, timeout_sec=10)


def sit(robot, command_client):
    robot.logger.info('Sitting down')
    blocking_sit(command_client, timeout_sec=10)


MENU = [
    ('Sit', sit),
    ('Stand', stand),
]


def print_menu():
    print()
    for i, (label, _) in enumerate(MENU, start=1):
        print(f'  {i}. {label}')
    print('  0. Quit')


def prompt_choice():
    print_menu()
    choice = input('> ').strip()
    if choice == '0':
        return None
    try:
        index = int(choice) - 1
        if 0 <= index < len(MENU):
            return MENU[index]
    except ValueError:
        pass
    print('invalid choice')
    return prompt_choice()


def main():
    sdk = bosdyn.client.create_standard_sdk('SpotBasicCommands')
    robot = sdk.create_robot(ROBOT_IP)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info('Powering on')
        robot.power_on(timeout_sec=20)

        try:
            while True:
                selection = prompt_choice()
                if selection is None:
                    break
                label, action = selection
                print(f'running: {label}')
                action(robot, command_client)
        finally:
            robot.logger.info('Powering off')
            safe_power_off(command_client, state_client)


if __name__ == '__main__':
    main()
