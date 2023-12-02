import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pygame
import sys

data = pd.read_csv('data/train.csv') #read data as pandas dataframe

data = np.array(data) #transform data into numpy array
np.random.shuffle(data) #shuffle data before splitting into training and testing datasets
m, n = data.shape #record the dimensions of the matrix

data_test = data[1:1000].T #use the first 1500 observations as test set, 
#then transpose the matrix such that observations are the columns and variables (pixels) are the rows
y_test = data_test[0] #row vector
x_test = data_test[1:n] #matrix with n-1 predictors
x_test = x_test / 255. #divide values by 255 so that they in [0,1]

#and similarly for the training data
data_train = data[1000:m].T
y_train = data_train[0] #row vector
x_train = data_train[1:n] #matrix with n-1 predictors
x_train = x_train / 255. #divide values by 255 so that they in [0,1]

def predict_digit(image, W1, b1, W2, b2): #used for pygame
    flattened_image = image.flatten() / 255.0 #flatten pygame image to vector of pixel luminances
    input_image = flattened_image.reshape((784, 1))
    predictions = make_predictions(input_image, W1, b1, W2, b2) #make predictions
    return predictions[0]

def init_params(): #function to initialize parameters, i.e. randomly assign initial weights and biases
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5 
    W2 = np.random.rand(10, 10) - 0.5 
    b2 = np.random.rand(10, 1) - 0.5 
    return W1, b1, W2, b2

def ReLU(Z): #defines first activation function ReLU
    return np.maximum(Z,0)

def softmax(Z): #defines second activation function softmax
    A = np.exp(Z)/ sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, x): #using activation functinos, function to handle forwatd prop
    Z1 = W1.dot(x) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z): #defines ReLU derivative functino
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((int(Y.size), int(Y.max() + 1)))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y): #defines backward propagation
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): #function to update parameters
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2): #returns prediction (highest likelihood)
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y): #returns accuracy of the model given true labels
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations): #function to handle gradient descent
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2): #function to make predictions
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2): 
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Drawing tool functions
def initialize_grid():
    return [[0 for _ in range(28)] for _ in range(28)]

def draw_grid(screen, grid):
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            pygame.draw.rect(screen, (color, color, color), (x * 20, y * 20, 20, 20))

def save_parameters(W1, b1, W2, b2, filename='parameters.npz'):
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)

def load_parameters(filename='parameters.npz'):
    data = np.load(filename)
    return data['W1'], data['b1'], data['W2'], data['b2']

def display_text(screen, text, position, font_size=36, fade_duration=2):
    font = pygame.font.SysFont("Courier New", font_size)
    text_surface = font.render(text, True, (255, 255, 255))

    screen.blit(text_surface, position)

def start_screen():
    pygame.init()
    screen = pygame.display.set_mode((560, 560))
    pygame.display.set_caption("Digit Recognizer - Start Screen")

    font = pygame.font.SysFont("Courier New", 36)
    titlefont = pygame.font.SysFont("Courier New Bold", 50)
    rulesfont = pygame.font.SysFont("Courier New", 16)

    button_use_saved = pygame.Rect(100, 250, 350, 50)
    button_compute_new = pygame.Rect(100, 350, 350, 50)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_use_saved.collidepoint(event.pos):
                    return 'saved'
                elif button_compute_new.collidepoint(event.pos):
                    iterations = compute_iterations()
                    return 'compute', iterations  # Return a tuple directly

        screen.fill((237, 225, 192))  # blue background

        pygame.draw.rect(screen, (255, 255, 255), button_use_saved, border_radius=5)  # Green button for using saved parameters
        pygame.draw.rect(screen, (255, 255, 255), button_compute_new, border_radius=5)  # Blue button for computing new parameters

        title = titlefont.render("MNIST Digit Recognizer", True, (0,0,0))
        text_saved = font.render("Load Parameters", True, (0, 0, 0))
        text_compute = font.render("New Parameters", True, (0, 0, 0))
        rules = rulesfont.render("Press C to clear    |    Press P to make a prediction",True, (0, 0, 0))

        screen.blit(title, (80, 80))
        screen.blit(text_saved, (button_use_saved.x + 10, button_use_saved.y + 10))
        screen.blit(text_compute, (button_compute_new.x + 20, button_compute_new.y + 10))
        screen.blit(rules, (10,535))

        pygame.display.flip()

# Define the compute_iterations function without an input box
def compute_iterations():
    screen = pygame.display.set_mode((560, 560))
    pygame.display.set_caption("Digit Recognizer - Number of Iterations")

    titlefont = pygame.font.SysFont("Courier New Bold", 40)
    font = pygame.font.Font(None, 36)

    input_text = ""
    input_rect = pygame.Rect(150, 200, 400, 50)
    active = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
            elif event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        try:
                            iterations = int(input_text)
                            return iterations  # Return the number of iterations
                        except ValueError:
                            print("Invalid input. Please enter a valid integer.")
                            input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode

        screen.fill((237, 225, 192))  # White background

        title = titlefont.render("Input the number of", True, (0,0,0))
        title2 = titlefont.render("propagation iterations", True, (0,0,0))

        pygame.draw.rect(screen, (0, 0, 0), input_rect, 2)
        text_surface = font.render(input_text, True, (0, 0, 0))
        width = max(200, text_surface.get_width() + 10)
        input_rect.w = width
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
        screen.blit(title, (20, 20))
        screen.blit(title2, (20, 50))

        pygame.display.flip()

def main():
    pygame.init()

    start_option = start_screen()

    if start_option == 'saved':
        try:
            W1, b1, W2, b2 = load_parameters()
        except FileNotFoundError:
            print("Saved parameters not found. Exiting.")
            sys.exit()
    elif isinstance(start_option, tuple) and start_option[0] == 'compute':
        iterations = start_option[1]
        W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, iterations)
        save_parameters(W1, b1, W2, b2)
    else:
        print("Invalid start option. Exiting.")
        sys.exit()

    screen = pygame.display.set_mode((560, 560))
    pygame.display.set_caption("Digit Recognizer")

    grid = initialize_grid()

    drawing = False
    prediction = None
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True  # Start drawing when the mouse button is pressed
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    x, y = pygame.mouse.get_pos() #store position of the mouse as x and y
                    if grid[y // 20][x // 20] != 255: #if pixel is not white
                        grid[y // 20][x // 20] = 255  # Set clicked pixel to whitecc
                        for i in range(-1, 2): #set adjacent pixel's to grey
                            for j in range(-1, 2):
                                if 0 <= (y // 20) + i < 28 and 0 <= (x // 20) + j < 28 and abs(i) + abs(j) == 1:
                                    if grid[(y // 20) + i][(x // 20) + j] == 0:
                                        grid[(y // 20) + i][(x // 20) + j] = 128
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False  # Stop drawing when the mouse button is released
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    # Clear the grid
                    grid = initialize_grid()
                elif event.key == pygame.K_p:
                    # Predict the digit
                    drawn_image = np.array(grid)
                    prediction = predict_digit(drawn_image, W1, b1, W2, b2)

        draw_grid(screen, grid)

        if prediction is not None:
            display_text(screen, f"Prediction: {prediction}", (10, 10))

        pygame.display.flip()

if __name__ == "__main__":
    main()