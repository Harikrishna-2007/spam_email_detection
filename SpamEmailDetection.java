import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

/**
 * Spam Email Detection System
 * ---------------------------
 * This program uses Machine Learning (Naive Bayes)
 * to classify emails as Spam or Ham using WEKA.
 *
 * Author: Your Name
 */

public class SpamEmailDetection {

    public static void main(String[] args) {
        try {

            // Load dataset
            DataSource source = new DataSource("data/spam.arff");
            Instances dataset = source.getDataSet();

            // Set class attribute
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }

            // Create Naive Bayes classifier
            NaiveBayes classifier = new NaiveBayes();

            // Train model
            classifier.buildClassifier(dataset);

            // Evaluate model using 10-fold cross validation
            Evaluation evaluation = new Evaluation(dataset);
            evaluation.crossValidateModel(classifier, dataset, 10, new Random(1));

            // Print results
            System.out.println("=================================");
            System.out.println(" Spam Email Detection Results ");
            System.out.println("=================================");
            System.out.println(evaluation.toSummaryString());
            System.out.println("Confusion Matrix:");
            System.out.println(evaluation.toMatrixString());
            System.out.println("Accuracy: " + evaluation.pctCorrect() + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
