from pandas import concat, DataFrame
from matplotlib.pyplot import legend, title, show

def plot_data(y_true, y_pred, title_name):
    draw = concat([DataFrame(y_true), DataFrame(y_pred)], axis=1)
    draw.iloc[:, 0].plot(figsize=(12, 6))
    draw.iloc[:, 1].plot(figsize=(12, 6))
    legend(('real', 'predict'), loc='upper right', fontsize='15')
    title(str(title_name), fontsize='30')
    show()
