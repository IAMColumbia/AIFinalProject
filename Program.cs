using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;

namespace simple_neural_network
{
    class NeuralNetWork
    {
        private Random _radomObj;

        public NeuralNetWork(int synapseMatrixColumns, int synapseMatrixLines)
        {
            SynapseMatrixColumns = synapseMatrixColumns;
            SynapseMatrixLines = synapseMatrixLines;

            _Init();
        }

        public int SynapseMatrixColumns { get; }
        public int SynapseMatrixLines { get; }
        public double[,] SynapsesMatrix { get; private set; }

        /// <summary>
        /// Initialize the ramdom object and the matrix of ramdon weights
        /// </summary>
        private void _Init()
        {
            // make sure that for every instance of the neural network we are geting the same radom values
            _radomObj = new Random(1);
            _GenerateSynapsesMatrix();
        }

        /// <summary>
        /// Generate our matrix with the weight of the synapses
        /// </summary>
        private void _GenerateSynapsesMatrix()
        {
            SynapsesMatrix = new double[SynapseMatrixLines, SynapseMatrixColumns];

            for (var i = 0; i < SynapseMatrixLines; i++)
            {
                for (var j = 0; j < SynapseMatrixColumns; j++)
                {
                    SynapsesMatrix[i, j] = (2 * _radomObj.NextDouble()) - 1;
                }
            }
        }

        /// <summary>
        /// Calculate the sigmoid of a value
        /// </summary>
        /// <returns></returns>
        private double[,] _CalculateSigmoid(double[,] matrix)
        {

            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    var value = matrix[i, j];
                    matrix[i, j] = 1 / (1 + Math.Exp(value * -1));
                }
            }
            return matrix;
        }

        /// <summary>
        /// Calculate the sigmoid derivative of a value
        /// </summary>
        /// <returns></returns>
        private double[,] _CalculateSigmoidDerivative(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    var value = matrix[i, j];
                    matrix[i, j] = value * (1 - value);
                }
            }
            return matrix;
        }

        /// <summary>
        /// Will return the outputs give the set of the inputs
        /// </summary>
        public double[,] Think(double[,] inputMatrix)
        {
            var productOfTheInputsAndWeights = MatrixDotProduct(inputMatrix, SynapsesMatrix);

            return _CalculateSigmoid(productOfTheInputsAndWeights);

        }

        /// <summary>
        /// Train the neural network to achieve the output matrix values
        /// </summary>
        public void Train(double[,] trainInputMatrix, double[,] trainOutputMatrix, int interactions)
        {
            // we run all the interactions
            for (var i = 0; i < interactions; i++)
            {
                // calculate the output
                var output = Think(trainInputMatrix);

                // calculate the error
                var error = MatrixSubstract(trainOutputMatrix, output);
                var curSigmoidDerivative = _CalculateSigmoidDerivative(output);
                var error_SigmoidDerivative = MatrixProduct(error, curSigmoidDerivative);

                // calculate the adjustment :) 
                var adjustment = MatrixDotProduct(MatrixTranspose(trainInputMatrix), error_SigmoidDerivative);

                SynapsesMatrix = MatrixSum(SynapsesMatrix, adjustment);
            }
        }

        /// <summary>
        /// Transpose a matrix
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixTranspose(double[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            double[,] result = new double[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Sum one matrix with another
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixSum(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] + matrixb[i, u];
                }
            }

            return result;
        }

        /// <summary>
        /// Subtract one matrix from another
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixSubstract(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] - matrixb[i, u];
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplication of a matrix
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixProduct(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] * matrixb[i, u];
                }
            }

            return result;
        }

        /// <summary>
        /// Dot Multiplication of a matrix
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixDotProduct(double[,] matrixa, double[,] matrixb)
        {

            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var rowsB = matrixb.GetLength(0);
            var colsB = matrixb.GetLength(1);

            if (colsA != rowsB)
                throw new Exception("Matrices dimensions don't fit.");

            var result = new double[rowsA, colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    for (int k = 0; k < rowsB; k++)
                        result[i, j] += matrixa[i, k] * matrixb[k, j];
                }
            }
            return result;
        }

    }

    class Program
    {

        static void PrintMatrix(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine);
            }
        }

        enum Traits
        {
            Health,
            Armor,
            Speed,
            Strength,
            Magic,
            Intelligence
        }


        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;

            var curNeuralNetwork = new NeuralNetWork(1, 6);

            Console.WriteLine("Synaptic weights before training:");
            PrintMatrix(curNeuralNetwork.SynapsesMatrix);

            //TODO : set up CreatureWithOutcome and convert to trainingInputs / outputs
            CreatureAndOutcome[] creatureAndOutcomes = new CreatureAndOutcome[]
            {
                new CreatureAndOutcome(.8, .5, .25, .9, 0, .1, .8), //Giant
                new CreatureAndOutcome(.5, .6, .4, .6, 0, .4, .7), //Knight
                new CreatureAndOutcome(0, 0, .75, 0, 0, .2, .05), //Rat
                new CreatureAndOutcome(.2, 0, .5, .2, 1, .9, .8), //Wizard
                new CreatureAndOutcome(0, 0, .5, 0, .5, 0, .1), //Ghost
                new CreatureAndOutcome(.9, .7, .25, .45, 0, .7, .8), //Tank
                new CreatureAndOutcome(.1, .3, .85, .4, 0, .65, .6) //Theif
            };

            //var trainingInputs = new double[,] { { 0, 0, 1 }, { 1, 1, 1 }, { 1, 0, 1 }, { 0, 1, 1 } };
            //var trainingOutputs = NeuralNetWork.MatrixTranspose(new double[,] { { 0, 1, 1, 0 } });

            double[,] trainingInputs = new double[creatureAndOutcomes.Length, 6];
            double[,] trainingOutputs = new double[1, creatureAndOutcomes.Length];

            CreatureAndOutcome creatureAndOutcome;
            for(int i = 0; i < creatureAndOutcomes.Length; i++)
            {
                creatureAndOutcome = creatureAndOutcomes[i];
                for(int j = 0; j < 6; j++)
                {
                    trainingInputs[i, j] = creatureAndOutcome.Traits[j];
                }
                trainingOutputs[0, i] = creatureAndOutcome.Outcome;
            }
            trainingOutputs = NeuralNetWork.MatrixTranspose(trainingOutputs);

            curNeuralNetwork.Train(trainingInputs, trainingOutputs, 10000);

            Console.WriteLine("\nSynaptic weights after training:");
            PrintMatrix(curNeuralNetwork.SynapsesMatrix);


            // testing neural networks against a new problem 
            //var output = curNeuralNetwork.Think(new double[,] { { 1, 0, 0, 1, .2, .5 } });
            //Console.WriteLine("\nConsidering new problem [1, 0, 0] => :");
            //PrintMatrix(output);

            Random rng = new Random();
            while(true)
            {
                Console.ForegroundColor = ConsoleColor.White;
                Creature player = new Creature(rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), rng.NextDouble());
                Creature enemy = new Creature(rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), rng.NextDouble());
                Console.WriteLine($"\n    Player Stats:\n{player.ToString()}");
                Console.WriteLine($"\n    Enemy Stats:\n{enemy.ToString()}");
                double playerOutcome = curNeuralNetwork.Think(player.Matrix)[0,0];
                double enemyOutcome = curNeuralNetwork.Think(enemy.Matrix)[0, 0];

                Console.WriteLine($"Player Outcome: {playerOutcome} | Enemy Outcome: {enemyOutcome}");
                Console.WriteLine("");
                bool playerWins = playerOutcome > enemyOutcome;
                Console.ForegroundColor = playerWins ? ConsoleColor.Green : ConsoleColor.Red;
                Console.WriteLine(playerWins ? "Player Wins" : "Enemy Wins");
                Console.WriteLine("");
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("Press Enter to Continue");
                Console.Read();
                Console.Clear();
            }
        }
    }

    public class Creature
    {
        public double[] Traits = new double[6];
        public double[,] Matrix { get { return new double[,] { { Traits[0], Traits[1], Traits[2], Traits[3], Traits[4], Traits[5] } }; } }

        public Creature(double health, double armor, double speed, double strength, double magic, double intelligence)
        {
            Traits[0] = health;
            Traits[1] = armor;
            Traits[2] = speed;
            Traits[3] = strength;
            Traits[4] = magic;
            Traits[5] = intelligence;
        }

        public override string ToString()
        {
            return $"Health: {Traits[0]}\nArmor: {Traits[1]}\nSpeed: {Traits[2]}\nStrength: {Traits[3]}\nMagic: {Traits[4]}\nIntelligence: {Traits[5]}";
        }
    }

    public class CreatureAndOutcome : Creature
    {
        public double Outcome { get; private set; }
        
        public CreatureAndOutcome(double health, double armor, double speed, double strength, double magic, double intelligence, double outcome) : base(health, armor, speed, strength, magic, intelligence)
        {
            Outcome = outcome;
        }
    }
}
