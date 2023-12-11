import numpy as np
def get_gram(price: list | np.ndarray):
    assert len(price) >= 3
    if type(price) == list:
        price = np.array(price)
    price1 = price[1:] - price[0]
    if price1[0] == 0:
        return price1[1:]
    else:
        return price1[1:] / price1[0]


def find_gram_index(dataframe_gram, gram):
    for i in range(len(dataframe_gram) - 5):
        if dataframe_gram['GRAM'][i] == gram:
            return i
    return -1
