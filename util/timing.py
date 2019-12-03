
import time
import math
import datetime

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % (as_minutes(s))


def time_since_and_expected_remaining_time(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- "expected time remaining: %s)' % (as_minutes(s), as_minutes(rs))


def date_time_now():
    return datetime.datetime.now()


def seconds_since_static(time_start: datetime, time_end: datetime):
    c = time_end - time_start
    return c.total_seconds()


def seconds_since(since: datetime):
    time_end = datetime.datetime.now()
    return seconds_since_static(since, time_end)


def milliseconds_since_static(time_start: datetime, time_end: datetime):
    c = time_end - time_start
    return c.total_seconds() * 1000


def milliseconds_since(since: datetime):
    time_end = datetime.datetime.now()
    return milliseconds_since_static(since, time_end) * 1000
