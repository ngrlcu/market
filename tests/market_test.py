import market.market as mk

def test_evolve():
    hist = mk.evolve(0, 100)
    assert(hist[0] == 0)
    assert(len(hist) == 101)
    
    hist = mk.evolve(2, 10)
    assert(hist[0] == 2)
    assert(len(hist) == 11)

def test_count_states():
    tmp = [1, 2, 1, 1, 1, 5, 0, 0, 0, 0] # 4 zeros, 4 ones, 1 two and 1 five (to be ignored)
    d = mk.count_states(tmp)
    assert(d[0]==4)
    assert(d[1]==4)
    assert(d[2]==1)
    assert(5 not in d)
