# cython: language_level=3

cdef enum DayCountConvention:
    BUS_252 = 0     # B3 / ANBIMA
    ACT_360 = 1     # US Money Market / SOFR
    ACT_365 = 2     # UK / Money Market
    THIRTY_360 = 3  # Corporate Bonds US/EU
    ACT_ACT = 4     # US Treasuries
    # Adicionais
    BUS_252_COLOMBIA = 5 # Exemplo de customização de paí­ses
    BUS_252_MEXICO = 6

cdef enum CompoundingType:
    EXPONENTIAL_B3 = 0  # (1 + r)^(DU/252)
    SIMPLE_LINEAR = 1   # 1 + r * (DC/360)
    COMPOUND_ANNUAL = 2 # (1 + r/k)^(k*t)
    CONTINUOUS = 3      # exp(r * t)
