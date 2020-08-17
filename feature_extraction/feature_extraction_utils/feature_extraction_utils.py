from PIL import Image
import imghdr


# Function to load an image from a path
def open_img(filename):
    img = Image.open(filename)
    return img


# Verify if a given image is using a valid format
def verify_valid_img(path):
    possible_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']
    if imghdr.what(path) in possible_formats:
        return True
    else:
        return False

# Columns of the data_frame
def create_columns(column_number, property):
    columns = []
    for i in range(0, column_number):
        columns.append(str(i))

    columns.append(property)
    return columns






