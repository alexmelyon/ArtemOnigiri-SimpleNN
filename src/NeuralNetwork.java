import java.util.function.UnaryOperator;

public class NeuralNetwork {

    private double learningRate;
    private Layer[] layers;
    private UnaryOperator<Double> activation;
    private UnaryOperator<Double> derivative;

    public NeuralNetwork(double learningRate, UnaryOperator<Double> activation, UnaryOperator<Double> derivative, int... sizes) {
        this.learningRate = learningRate;
        this.activation = activation;
        this.derivative = derivative;
        layers = new Layer[sizes.length];
        for (int layerNum = 0; layerNum < sizes.length; layerNum++) {
            int nextSize = 0;
            if(layerNum < sizes.length - 1) {
                nextSize = sizes[layerNum + 1];
            }
            layers[layerNum] = new Layer(sizes[layerNum], nextSize);
            for (int layerCount = 0; layerCount < sizes[layerNum]; layerCount++) {
                layers[layerNum].biases[layerCount] = Math.random() * 2.0 - 1.0;
                for (int nextCount = 0; nextCount < nextSize; nextCount++) {
                    layers[layerNum].weights[layerCount][nextCount] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int i = 1; i < layers.length; i++)  {
            Layer l = layers[i - 1];
            Layer l1 = layers[i];
            for (int j = 0; j < l1.size; j++) {
                l1.neurons[j] = 0;
                for (int k = 0; k < l.size; k++) {
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                }
                l1.neurons[j] += l1.biases[j];
                l1.neurons[j] = activation.apply(l1.neurons[j]);
            }
        }
        return layers[layers.length - 1].neurons;
    }

    public void backpropagation(double[] targets) {
        int lastButOneLayerNeuronsCount = layers[layers.length - 1].size;
        double[] errors = new double[lastButOneLayerNeuronsCount];
        for (int i = 0; i < lastButOneLayerNeuronsCount; i++) {
            errors[i] = targets[i] - layers[layers.length - 1].neurons[i];
        }
        for (int layerNum = layers.length - 2; layerNum >= 0; layerNum--) {
            Layer layer = layers[layerNum];
            Layer layerNext = layers[layerNum + 1];
            double[] errorsNext = new double[layer.size];
            double[] gradients = new double[layerNext.size];
            for (int i = 0; i < layerNext.size; i++) {
                gradients[i] = errors[i] * derivative.apply(layers[layerNum + 1].neurons[i]);
                gradients[i] *= learningRate;
            }
            double[][] deltas = new double[layerNext.size][layer.size];
            for (int i = 0; i < layerNext.size; i++) {
                for (int j = 0; j < layer.size; j++) {
                    deltas[i][j] = gradients[i] * layer.neurons[j];
                }
            }
            for (int i = 0; i < layer.size; i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < layerNext.size; j++) {
                    errorsNext[i] += layer.weights[i][j] * errors[j];
                }
            }
            errors = new double[layer.size];
            System.arraycopy(errorsNext, 0, errors, 0, layer.size);
            double[][] weightsNew = new double[layer.weights.length][layer.weights[0].length];
            for (int i = 0; i < layerNext.size; i++) {
                for (int j = 0; j < layer.size; j++) {
                    weightsNew[j][i] = layer.weights[j][i] + deltas[i][j];
                }
            }
            layer.weights = weightsNew;
            for (int i = 0; i < layerNext.size; i++) {
                layerNext.biases[i] += gradients[i];
            }
        }
    }

}