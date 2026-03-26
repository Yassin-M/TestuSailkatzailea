package sailkatzailea;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.MultiNomialBMAEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

/**
 * BayesNet sailkatzailearen parametroen fine-tuning prozesua kudeatzen du.
 * Estimazio-metodoak, bilaketa-algoritmoak, maxParents kopurua eta
 * alpha balioak optimizatzen ditu Grid Search bidez.
 */
public class BayesNetFineTuning {

    // SINGLETON
    private static BayesNetFineTuning nireBNFT;
    private BayesNetFineTuning() {}
    public static BayesNetFineTuning getFineTuning() {
        if (nireBNFT == null) {
            nireBNFT = new BayesNetFineTuning();
        }
        return nireBNFT;
    }

    /**
     * BayesNet ereduaren fine-tuning prozesua exekutatzen du, emandako datu-multzoa erabiliz.
     * Hainbat bilaketa-algoritmo, estimatzaile, maxParents eta alpha konbinazio probatzen ditu
     * 10-fold cross-validation erabiliz.
     *
     * @param datuBektorizatuak Weka DataSource objektua bektorizatutako datuekin.
     * @return BayesNet objektua, aurkitutako parametro optimoekin erabat konfiguratuta.
     * @throws Exception Datuak irakurtzean, ebaluatzean edo eredua kopiatzean erroreren bat gertatzen bada.
     */
    public BayesNet fineTune(DataSource datuBektorizatuak) throws Exception {
        Instances data = datuBektorizatuak.getDataSet();

        // KLASEA EZARRI (AZKEN ATRIBUTUA?)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        System.out.println("BayesNet-en fine-tuning prozesua abiarazten...\n");

        // PARAMETROAK
        BayesNetEstimator[] estimatzaileak = {
                new BayesNetEstimator(),
                new BMAEstimator(),
                new SimpleEstimator(),
                new MultiNomialBMAEstimator()
        };

        LocalScoreSearchAlgorithm[] bilaketaAlgoritmoak = {
                new GeneticSearch(),
                new HillClimber(),
                new K2(),
                new LAGDHillClimber(),
                new RepeatedHillClimber(),
                new SimulatedAnnealing(),
                new TabuSearch(),
                new TAN()
        };

        int[] maxParentsValues = {1, 2};
        double[] alphaValues = {0.1, 0.5, 1.0};

        double bestFMeasure = -1.0;
        String bestConfig = "";
        BayesNet bestBayesNet = null;
        int i = 1;

        // Parametro ekorketa
        for (LocalScoreSearchAlgorithm searchAlgo : bilaketaAlgoritmoak) {
            for (BayesNetEstimator estimator : estimatzaileak) {
                for (int maxParents : maxParentsValues) {
                    for (double alpha : alphaValues) {
                        System.out.println("Iterazioa: " + i);

                        BayesNet bayesNet = getBayesNet(searchAlgo, estimator, maxParents, alpha);

                        // 10-fold Cross-Validation
                        Evaluation eval = new Evaluation(data);
                        eval.crossValidateModel(bayesNet, data, 10, new Random(1));

                        // Metrikak atera
                        double fMeasure = eval.weightedFMeasure();
                        double accuracy = eval.pctCorrect();

                        String algoName = searchAlgo.getClass().getSimpleName();
                        String estimatorName = estimator.getClass().getSimpleName();

                        System.out.printf("Parametroak -> Algoritmoa: %s | Estimatzailea: %s | maxParents: %d | Alpha: %.1f%n",
                                algoName, estimatorName, maxParents, alpha);
                        System.out.printf("Emaitza -> Accuracy: %.2f%% | Weighted F-Measure: %.4f%n", accuracy, fMeasure);
                        System.out.println("---------------------------------------------------");

                        // Konfigurazio hobea bada, erregistroa eguneratu
                        if (fMeasure > bestFMeasure) {
                            bestFMeasure = fMeasure;
                            bestConfig = String.format("Algoritmoa=%s, Estimatzailea=%s, maxParents=%d, alpha=%.1f",
                                    algoName, estimatorName, maxParents, alpha);

                            // Ereduaren kopia sakona egin
                            bestBayesNet = (BayesNet) AbstractClassifier.makeCopy(bayesNet);
                        }
                        i++;
                    }
                }
            }
        }

        // Emaitza
        System.out.println("\n=== EMAITZA OPTIMOA ===");
        System.out.println("Konfigurazio hoberena: " + bestConfig);
        System.out.println("Weighted F-Measure estimatua: " + bestFMeasure);

        return bestBayesNet;
    }

    /**
     * BayesNet sailkatzailearen instantzia bat sortzen eta konfiguratzen du emandako espezifikazioekin.
     *
     * @param searchAlgo Erabili beharreko bilaketa-algoritmoa (LocalScoreSearchAlgorithm motakoa).
     * @param estimator Erabili beharreko sare bayestarraren estimatzailea.
     * @param maxParents Guraso nodoen kopuru maximoa (algoritmoaren konplexutasuna mugatzeko).
     * @param alpha Alpha parametroa estimatzailearentzat.
     * @return BayesNet objektua bere parametro guztiekin konfiguratuta.
     */
    private static BayesNet getBayesNet(LocalScoreSearchAlgorithm searchAlgo, BayesNetEstimator estimator, int maxParents, double alpha) {
        BayesNet bayesNet = new BayesNet();

        // Parametro hauek bakarrik K2 eta HillClimber algoritmoetan daude
        if (searchAlgo instanceof K2) {
            ((K2) searchAlgo).setMaxNrOfParents(maxParents);
            ((K2) searchAlgo).setInitAsNaiveBayes(true);
        } else if (searchAlgo instanceof HillClimber) {
            ((HillClimber) searchAlgo).setMaxNrOfParents(maxParents);
            ((HillClimber) searchAlgo).setInitAsNaiveBayes(true);
        }

        bayesNet.setSearchAlgorithm(searchAlgo);

        estimator.setAlpha(alpha);
        bayesNet.setEstimator(estimator);

        return bayesNet;
    }
}