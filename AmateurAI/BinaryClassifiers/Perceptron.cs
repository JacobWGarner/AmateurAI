using System;

namespace AmateurAI.BinaryClassifiers
{
    /// <summary>
    /// Represents an artificial neuron that can learn to classify between 
    /// two diffirent types of data after training.
    /// </summary>
    public class Perceptron
    {
        #region Properties

        /// <summary>
        /// Gets or sets the learning rate, which determines how quickly the 
        /// perceptron "learns".
        /// </summary>
        public float LearningRate { get; set; }

        /// <summary>
        /// Gets the weights associated with each connection to the perceptron.
        /// </summary>
        public float[] Weights { get; private set; }

        /// <summary>
        /// Gets the bias, which determines when the perceptron becomes 
        /// "meaningfully active".
        /// </summary>
        public float Bias { get; private set; }

        /// <summary>
        /// Gets the size, which is how many connections the perceptron has.
        /// </summary>
        public int Size { get; private set; }

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Perceptron"/> class.
        /// </summary>
        /// <param name="size">The number of connections the perceptron will 
        /// have.</param>
        /// <param name="learningRate">The rate of how fast the perceptron 
        /// will "learn".</param>
        /// <param name="rand">Used to generate random weights and a bias for 
        /// the perceptron.</param>
        public Perceptron(int size, float learningRate, Random rand)
        {
            //Check that the arguments passed are valid.
            if (size > 0 && learningRate > 0)
            {
                //Initialize the size, bias, and learning rate.
                this.Size = size;
                this.LearningRate = learningRate;
                this.Bias = SmallRandomFloat(rand);

                //Initialize the weights with small random values.
                this.Weights = new float[size];

                for (int i = 0; i < Size; i++)
                {
                    this.Weights[i] = SmallRandomFloat(rand);
                }
            }

            else
            {
                throw new Exception("The size and learningRate must both be " +
                    "greater than 0.");
            }
        }

        #endregion

        #region Functions

        /// <summary>
        /// Evaluates input data and classifies it.
        /// </summary>
        /// <param name="input">The input data passed to the perceptron for 
        /// evaluation.</param>
        /// <returns>The output of the perceptron, either 0 or 1.</returns>
        public float Predict(float[] input)
        {
            //Check that the arguments passed are valid.
            if (input.Length == Size)
            {
                //Calculate the raw output of the perceptron.
                float output = 0;

                for (int i = 0; i < Size; i++)
                {
                    output += input[i] * Weights[i];
                }

                output += Bias;

                //Run the output through the activation function and return
                //the result.
                if (output > 0)
                    return 1;

                else
                    return 0;
            }

            else
            {
                throw new Exception("The length of input (float[] input) " +
                    "and Size (Perceptron.Size) must match.");
            }
        }

        /// <summary>
        /// Trains the perceptron on a single training example.
        /// </summary>
        /// <param name="input">The input from the training example.</param>
        /// <param name="expectedOutput">The output expected from the 
        /// perceptron given training example input.</param>
        /// <returns>The absolute value of the error calculated after the 
        /// perceptron makes its prediction on the training example.</returns>
        public float Train(float[] input, float expectedOutput)
        {
            //Check that the arguments passed are valid.
            if (input.Length == Size)
            {
                //Classify the input and then calculate the error.
                float output = Predict(input);
                float error = expectedOutput - output;

                //Update the weights and bias.
                for (int i = 0; i < Size; i++)
                {
                    Weights[i] = Weights[i] + LearningRate * error * input[i];
                }

                Bias = Bias + LearningRate * error;

                return Math.Abs(error);
            }

            else
            {
                throw new Exception("The length of input (float[] input) " +
                    "and Size (Perceptron.Size) must match.");
            }
        }

        /// <summary>
        /// Trains the perceptron on a batch of training examples.
        /// </summary>
        /// <param name="inputs">The inputs from each training example.</param>
        /// <param name="expectedOutputs">The outputs expected from the 
        /// perceptron given each training example input.</param>
        /// <returns>The accuracy of the perceptron as a percent value (i.e. 
        /// 70, 30, etc.)</returns>
        public float TrainBatch(float[][] inputs, float[] expectedOutputs)
        {
            //Check that the arguments passed are valid.
            if (inputs[0].Length == Size)
            {
                float errorSum = 0;

                for (int i = 0; i < inputs.Length; i++)
                {
                    errorSum += Train(inputs[i], expectedOutputs[i]);
                }

                #region Programmer's note

                /*=================================================================
                 * The error sum is incremented by one every time the perceptron 
                 * gives a wrong answer to a training example. Therefore, the 
                 * formula for calculating the accuracy is:
                 * 
                 * (Number of correct answers / Number of training examples) * 100 
                 * 
                 * Number of correct answers = 
                 *      Number of training examples - number of wrong answers
                 *///==============================================================

                #endregion
                return ((inputs.Length - errorSum) / inputs.Length) * 100;
            }

            else
            {
                throw new Exception("The length of any input " +
                    "(float[] inputs) and Size (Perceptron.Size) must match.");
            }
        }

        /// <summary>
        /// Generates a small random value, ranging from -0.9 to 0.9.
        /// </summary>
        /// <param name="rand">Used to generate random values.</param>
        /// <returns>A float from -0.9 to 0.9.</returns>
        private float SmallRandomFloat(Random rand)
        {
            #region Programmer's note
            /*=================================================================
             * The top part of this equation randomly generates a value 
             * between 0.0 and 0.9. The second half has a 50% chance to 
             * multiply that value by either 1 or -1. This creates a random 
             * value between -0.9 and 0.9.
             *///==============================================================
            #endregion
            return (float)rand.NextDouble() 
                * (rand.NextDouble() >= 0.5 ? 1 : -1);
        }

        #endregion

        #region Sources

        /* Sources used while developing:
         * 
         * 3blue1brown - General conecpt and math for calculating the output: 
         * https://youtu.be/aircAruvnKk?t=519
         * 
         * Simplilearn - Activation function:
         * https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron
         * 
         * ritvikmath - Math for updating weights: 
         * https://youtu.be/4Gac5I64LM4?t=476
         * 
         * Jason Brownlee - Math for updating weights and bias: 
         * https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
         * https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/
         * 
         * Thank you to all the people who made these sources of information 
         * so that I could learn how to program a perceptron!
         * 
         * 1 Peter 4:10-11
         */

        #endregion
    }
}