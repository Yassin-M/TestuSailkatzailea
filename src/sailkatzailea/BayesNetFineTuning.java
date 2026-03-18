package sailkatzailea;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class BayesNetFineTuning {

    public void fineTune(DataSource datuBektorizatuak) throws Exception {
        Instances data = datuBektorizatuak.getDataSet();

        // KLASEA EZARRI (AZKEN ATRIBUTUA?)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        System.out.println("BayesNet-en fine-tuning prozesua abiarazten...\n");

        // PARAMETROAK
        int[] maxParentsValues = {1, 2};
        double[] alphaValues = {0.1, 0.5, 1.0};

        double bestFMeasure = -1.0;
        String bestConfig = "";

        // Parametro ekorketa
        for (int maxParents : maxParentsValues) {
            for (double alpha : alphaValues) {

                BayesNet bayesNet = getBayesNet(maxParents, alpha);

                // (10-fold Cross-Validation)
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(bayesNet, data, 10, new Random(1));

                // Metrikak atera
                double fMeasure = eval.weightedFMeasure();
                double accuracy = eval.pctCorrect();

                System.out.printf("Parametroak -> maxParents: %d | Alpha: %.1f%n", maxParents, alpha);
                System.out.printf("Emaitza -> Accuracy: %.2f%% | Weighted F-Measure: %.4f%n", accuracy, fMeasure);
                System.out.println("---------------------------------------------------");

                if (fMeasure > bestFMeasure) {
                    bestFMeasure = fMeasure;
                    bestConfig = "maxParents=" + maxParents + ", alpha=" + alpha;
                }
            }
        }

        // Emaitza
        System.out.println("\n--- EMAITZA OPTIMOA ---");
        System.out.println("Konfigurazio hoberena: " + bestConfig);
        System.out.println("Weighted F-Measure estimatua: " + bestFMeasure);
    }

    private static BayesNet getBayesNet(int maxParents, double alpha) {
        BayesNet bayesNet = new BayesNet();

        K2 searchAlgo = new K2();
        searchAlgo.setMaxNrOfParents(maxParents);
        searchAlgo.setInitAsNaiveBayes(true);
        bayesNet.setSearchAlgorithm(searchAlgo);

        SimpleEstimator estimator = new SimpleEstimator();
        estimator.setAlpha(alpha);
        bayesNet.setEstimator(estimator);
        return bayesNet;
    }
}