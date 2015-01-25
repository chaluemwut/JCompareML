package com.kku.jcompareml;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.RandomForest;
import weka.core.FastVector;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by off on 1/25/15.
 */
public class CompareML {

    public static Evaluation classify(Classifier model,
                                      Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);

        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }

    public static void main(String [] args){
        try {
            BufferedReader datafile = readDataFile("/home/off/IdeaProjects/JCompareML/data/letterp2.arff");
            Classifier cls = new RandomForest();
            Instances data = new Instances(datafile);
            data.setClassIndex(data.numAttributes() - 1);

            Instances[][] split = crossValidationSplit(data, 10);

            Instances[] trainingSplits = split[0];
            Instances[] testingSplits = split[1];

            FastVector predictions = new FastVector();

            // For each training-testing split pair, train and test the classifier
            for (int i = 0; i < trainingSplits.length; i++) {
                Evaluation validation = classify(cls, trainingSplits[i], testingSplits[i]);

                predictions.appendElements(validation.predictions());

                // Uncomment to see the summary for each training-testing pair.
                //System.out.println(models[j].toString());
            }

            // Calculate overall accuracy of current classifier on all splits
            double accuracy = calculateAccuracy(predictions);

            // Print current classifier's name and accuracy in a complicated,
            // but nice-looking way.
            System.out.println("Accuracy of " + cls.getClass().getSimpleName() + ": "
                    + String.format("%.2f%%", accuracy)
                    + "\n---------------------------------");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
