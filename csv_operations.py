# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Various operations eprformed on Comma Separated Value files
"""
import csv


def csv_write(filename, x, predicted, xname, yname, zname, y_actual):
    """
    Docstring for csv_write
    
    :param filename: Name of file exported
    :param x: X Values
    :param predicted: Y Values
    :param xname: Header for X values
    :param yname: Header for Y values
    :param zname: Header for Z values
    :param y_actual: Z values
    """
    with open(filename, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([xname, yname, zname])  # header
        for xi, yi, zi in zip(x, predicted, y_actual):
            writer.writerow([xi.item(), yi.item(), zi.item()])  # convert tensors to Python floats

def csv_write2(filename, x, predicted, xname, yname, zname, wname, y_actual, w_actual, filemode=0):
    """
    Docstring for csv_write2
    
    :param filename: Filename to be exported
    :param x: X values
    :param predicted: Y values
    :param xname: X header
    :param yname: Y header
    :param zname: Z header
    :param wname: W header
    :param y_actual: Y values
    :param w_actual: W values
    :param exec: file mode
    """
    if filemode == 0:
        mode = 'w'
    else:
        mode = 'a'
    with open(filename, mode=mode, newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        if filemode == 0:
            writer.writerow([xname, yname, zname, wname])  # header
        for xi, yi, zi, wi in zip(x, predicted, y_actual, w_actual):
            writer.writerow([xi.item(), yi.item(), zi.item(), wi.item()])

def csv_read(filename, xname, yname, zname):
    """
    Docstring for csv_read
    
    :param filename: File name to read
    :param xname: X header
    :param yname: Y header
    :param zname: Z header
    """
    x_vals = []
    y_preds = []
    z_preds = []
    reader = csv.DictReader(filename)  # read as dictionary
    with open(filename, 'r', encoding = 'utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(float(row[xname]))
            y_preds.append(float(row[yname]))
            z_preds.append(float(row[zname]))
    return x_vals, y_preds, z_preds

def csv_read2(filename, xname, yname, zname, wname):
    """
    Docstring for csv_read2
    
    :param filename: Filename to read
    :param xname: X header
    :param yname: Y header
    :param zname: Z header
    :param wname: W header
    """
    x_vals = []
    y_preds = []
    z_preds = []
    w_preds = []
    reader = csv.DictReader(filename)  # read as dictionary
    with open(filename, 'r', encoding = 'utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(float(row[xname]))
            y_preds.append(float(row[yname]))
            z_preds.append(float(row[zname]))
            w_preds.append(float(row[wname]))
    return x_vals, y_preds, z_preds, w_preds
