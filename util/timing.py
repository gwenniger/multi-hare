
import time
import math
import datetime


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


def date_time_start():
    return datetime.datetime.now()


def seconds_since(since: datetime):
    b = datetime.datetime.now()
    c = b - since
    return c.total_seconds()


def milliseconds_since(since: datetime):
    return seconds_since(since) * 1000
