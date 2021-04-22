# Copyright (c) 2021 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from datetime import datetime


def generate_datetime():
    """Generate datetime string

    Returns:
        str: datetime string
    """
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    return date_time
