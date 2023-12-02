\documentclass{article}

%page setup
\usepackage[english]{babel}
\usepackage[letterpaper,top=2cm,bottom=2cm,left=3cm,right=3cm,marginparwidth=1.75cm]{geometry}

% Useful packages
\usepackage{amsmath}
\usepackage{amssymb}
\newcommand{\RR}{\mathbb{R}}
\usepackage[colorlinks=true, allcolors=blue]{hyperref}
\usepackage{graphicx}
\usepackage{pgfplots}
\pgfplotsset{width=10cm,compat=1.9}
\usepackage{tikz}
\usepackage{listofitems}
\usetikzlibrary{arrows.meta}
\usepackage[outline]{contour}
\contourlength{1.4pt}

\tikzset{>=latex} % for LaTeX arrow head
\usepackage{xcolor}
\colorlet{myred}{red!80!black}
\colorlet{myblue}{blue!80!black}
\colorlet{mygreen}{green!60!black}
\colorlet{myorange}{orange!70!red!60!black}
\colorlet{mydarkred}{red!30!black}
\colorlet{mydarkblue}{blue!40!black}
\colorlet{mydarkgreen}{green!30!black}
\tikzstyle{node}=[thick,circle,draw=myblue,minimum size=22,inner sep=0.5,outer sep=0.6]
\tikzstyle{node in}=[node,green!20!black,draw=mygreen!30!black,fill=mygreen!25]
\tikzstyle{node hidden}=[node,blue!20!black,draw=myblue!30!black,fill=myblue!20]
\tikzstyle{node convol}=[node,orange!20!black,draw=myorange!30!black,fill=myorange!20]
\tikzstyle{node out}=[node,red!20!black,draw=myred!30!black,fill=myred!20]
\tikzstyle{connect}=[thick,mydarkblue] %,line cap=round
\tikzstyle{connect arrow}=[-{Latex[length=4,width=3.5]},thick,mydarkblue,shorten <=0.5,shorten >=1]
\tikzset{ % node styles, numbered for easy mapping with \nstyle
  node 1/.style={node in},
  node 2/.style={node hidden},
  node 3/.style={node out},
}
\def\nstyle{int(\lay<\Nnodlen?min(2,\lay):3)}

\title{Developing a Neural Network From Scratch}
\author{Elliott Chang}
\date{}

\begin{document}

\maketitle

\begin{abstract}
    This document presents a comprehensive overview of a neural network project aimed at recognizing and classifying handwritten digits using the MNIST dataset. The project involves the implementation of a neural network from scratch, providing insights into the code, methodology, and the underlying mathematical principles. Through this work, I explore the intricacies of training a model to perform image classification tasks, shedding light on the architecture, training process, and results achieved.
\end{abstract}

\newpage
\section{Introduction}

Handwritten digit recognition serves as a fundamental problem in the field of machine learning and artificial intelligence. The ability to develop a model capable of accurately identifying and classifying digits not only has practical applications in optical character recognition but also serves as a foundational exercise for understanding neural networks.

In this project, I delve into the MNIST dataset, a benchmark dataset widely used for image classification tasks. The dataset consists of a collection of 28x28 pixel grayscale images of handwritten digits (0 through 9), making it an ideal playground for training and evaluating machine learning models.

The focus of this endeavor is on implementing a neural network entirely from scratch. By doing so, I aim to provide a transparent view into the inner workings of the network, from the fundamental mathematical concepts to the code implementation. This document will guide the reader through the architecture of the neural network, the intricacies of training on the MNIST dataset, and the evaluation of the model's performance.

\section{The Math Behind the Madness}

%ADD SECTION EXPLAINING THE DIFFERENT PARTS INCLUDING FORWARD AND BACKWARD PROPAGATION AND UPDATING PARAMETERS

In the realm of neural networks, the underlying mathematics revolves around the concept of a perceptron, the basic building block of these models. A perceptron takes multiple input values, each multiplied by a corresponding weight, and sums these weighted inputs. The result undergoes an activation function, introducing non-linearity to the model. This process can be succinctly expressed as
$$\text{output} = \text{activation}\Bigl(\sum_i \text{input}_i \times \text{weight}_i + \text{bias}_i\Bigr).$$
Stacking multiple perceptrons in layers creates a neural network. The first layer, known as the input layer, receives the initial features of the data. Subsequent layers, referred to as hidden layers, perform intermediate computations, adjusting weights and applying activation functions. The final layer, the output layer, produces the model's predictions. This architecture is visualized in Figure 1. The connections between neurons, governed by weights, are adjusted during training using algorithms like gradient descent to minimize the difference between predicted and actual outputs. The backpropagation algorithm computes the gradient (derivative) of a loss function with respect to the weights, facilitating this optimization process. The mathematical foundation of neural networks thus lies in linear algebra, calculus, and the interplay between weights and activation functions, enabling these models to learn complex patterns and relationships in data. 

\begin{center}
\begin{tikzpicture}[x=2.2cm,y=1.4cm]
  \message{^^JNeural network, shifted}
  \readlist \Nnod{4,5,5,5,3} % array of number of nodes per layer
  \readlist\Nstr{n,m,m,m,k} % array of string number of nodes per layer
  \readlist\Cstr{\strut x,a^{(\prev)},a^{(\prev)},a^{(\prev)},y} % array of coefficient symbol per layer
  \def\yshift{0.5} % shift last node for dots
  
  \message{^^J  Layer}
  \foreachitem \N \in \Nnod{ % loop over layers
    \def\lay{\Ncnt} % alias of index of current layer
    \pgfmathsetmacro\prev{int(\Ncnt-1)} % number of previous layer
    \message{\lay,}
    \foreach \i [evaluate={\c=int(\i==\N); \y=\N/2-\i-\c*\yshift;
                 \index=(\i<\N?int(\i):"\Nstr[\lay]");
                 \x=\lay; \n=\nstyle;}] in {1,...,\N}{ % loop over nodes
      % NODES
      \node[node \n] (N\lay-\i) at (\x,\y) {$\Cstr[\lay]_{\index}$};
      
      % CONNECTIONS
      \ifnum\lay>1 % connect to previous layer
        \foreach \j in {1,...,\Nnod[\prev]}{ % loop over nodes in previous layer
          \draw[connect,white,line width=1.2] (N\prev-\j) -- (N\lay-\i);
          \draw[connect] (N\prev-\j) -- (N\lay-\i);
          %\draw[connect] (N\prev-\j.0) -- (N\lay-\i.180); % connect to left
        }
      \fi % else: nothing to connect first layer
      
    }
    \path (N\lay-\N) --++ (0,1+\yshift) node[midway,scale=1.5] {$\vdots$};
  }
  
  % LABELS
  \node[above=5,align=center,mygreen!60!black] at (N1-1.90) {input\\[-0.2em]layer};
  \node[above=2,align=center,myblue!60!black] at (N3-1.90) {hidden layers};
  \node[above=10,align=center,myred!60!black] at (N\Nnodlen-1.90) {output\\[-0.2em]layer};
  
\end{tikzpicture} \\
Figure 1: A neural network with $3$ hidden layers
\end{center}

For this project, I designed a straightforward neural network comprising a single hidden layer. The input layer, denoted as $X$, encompasses $784$ nodes corresponding to the $784$ pixels in each $28\times28$ pixel image. Each node represents the luminance of its corresponding pixel, ranging between $0$ and $1$. To enable digit predictions, the output layer requires $10$ nodes, each representing a possible digit ($0$-$9$). Consequently, the neural network processes $784$-dimensional input vectors to produce $10$-dimensional output vectors. To handle the entire dataset with $m$ observations, the network takes in a matrix of dimensions $784\times m$ (where each column represents an individual observation) and outputs a matrix of dimensions $10\times m$. This matrix-based approach efficiently allows the network to perform parallel computations on the entire dataset during forward and backward passes, optimizing the training process. I chose to incorporate a 10-node hidden layer. For the activation function in the hidden layer, I opted for the simple ReLU function, defined as $$\text{ReLU}(z_i) = \max(z_i,0).$$ The activation function guiding the transition from the inactivated output layer values to the activated output layer is the softmax function, defined as $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}},$$ which transforms values from the hidden layer into probabilities. The ensuing equations outline the network's computations:

\begin{align*}
&Z^{[1]} = W^{[1]}X + b^{[1]} \\
&A^{[1]} = \text{ReLU}(Z^{[1]}) \\
&Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
&A^{[2]} = \text{softmax}(Z^{[2]}).
\end{align*}

However, in order to effectively classify our inputs, we need to tune our weights and biases. This is done through backward propagation. 
Backward propagation is an iterative process that fine-tunes the model's weights and biases based on how much a given prediction deviated from the true label. After the forward propagation phase, where the input data passes through the network to generate predictions, the backpropagation algorithm is employed to calculate the gradient (derivative) of the loss with respect to each weight in the network. This involves traversing the network in the reverse direction, starting from the output layer and moving backward through the hidden layers to the input layer. The gradient is then used to update the weights, nudging them in the direction that reduces the overall error. This iterative cycle of forward and backward propagation continues until the model's performance converges to a desirable state. Using the chain rule, this process is able to identify how much each weight and bias contributed to the overall error. For our particular network, backward propagation is performed using the following equations:
\begin{align*}
    &dZ^{[2]} = A^{[2]}-Y \\
    &dW^{[2]} = \frac{1}{m}dZ^{[2]}(A^{[1]})^T \\
    &db^{[2]} = \frac{1}{m}\sum dZ^{[2]} \\
    &dZ^{[1]} = (W^{[2]})^T dZ^{[2]}*\text{ReLU}'(Z^{[1]}) \\
    &dW^{[1]} = \frac{1}{m}dZ^{[1]}(A^{[0]})^T \\
    &db^{[1]} = \frac{1}{m}\sum dZ^{[1]}.
\end{align*}
To see how these backward propagation equations are computed, watch this \href{https://www.youtube.com/watch?v=5-rVLSc2XdE}{video}. Based on these equations, we can update our parameters by
\begin{align*}
    &W^{[1]}=W^{[1]}-\alpha dW^{[1]}\\
    &b^{[1]}=b^{[1]}-\alpha db^{[1]}\\
    &W^{[2]}=W^{[2]}-\alpha dW^{[2]}\\
    &b^{[2]}=b^{[2]}-\alpha db^{[2]}
\end{align*}
where $\alpha$ is the learning rate of the model, the updated parameters are on the left and the previous parameters are on the right.

\section{Implementation in Python}

To see the implementation of this neural network in Python, visit the Jupyter Notebook I created. The neural network ended up reaching a training accuracy of about $85\%$. 

\end{document}
