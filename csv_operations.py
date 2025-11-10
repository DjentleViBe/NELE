import csv 

def csv_write(filename, x, predicted, xname, yname):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([xname, yname])  # header
        for xi, yi in zip(x, predicted):
            writer.writerow([xi.item(), yi.item()])  # convert tensors to Python floats
    
def csv_read(filename, xname, yname):
    x_vals = []
    y_preds = []
    reader = csv.DictReader(filename)  # read as dictionary
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(float(row[xname]))
            y_preds.append(float(row[yname]))
    return x_vals, y_preds
