package main.sailkatzailea;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.MultiNomialBMAEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/**
 * BayesNet sailkatzailearen parametroen fine-tuning prozesua.
 * Estimazio-metodoak, bilaketa-algoritmoak, maxParents kopurua eta
 * alpha balioak optimizatzen ditu.
 */
public class BayesNetFineTuning {

    private Instances train;
    private Instances test;

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
     * @param pTrain Weka Instances objektua, bektorizatutako datuekin.
     * @throws Exception Datuak irakurtzean, ebaluatzean edo eredua kopiatzean erroreren bat gertatzen bada.
     */
    public void fineTune(Instances pTrain, Instances pTest) throws Exception {
        train = pTrain;
        test = pTest;

        // KLASEA EZARRI (AZKEN ATRIBUTUA?)
        if(train.classIndex() == -1) train.setClassIndex(train.attribute("Sentiment").index());
        if(test.classIndex() == -1) test.setClassIndex(test.attribute("Sentiment").index());

        txikituDatuak();

        System.out.println("BayesNet-en fine-tuning prozesua abiarazten...\n");

        // PARAMETROAK
        BayesNetEstimator[] estimatzaileak = {
                new BMAEstimator(),
                new SimpleEstimator(),
                new MultiNomialBMAEstimator()
        };

        LocalScoreSearchAlgorithm[] bilaketaAlgoritmoak = {
                new HillClimber(),
                new K2(),
                new TAN()
        };

        int[] maxParentsValues = {1, 2};
        double[] alphaValues = {0.5, 1.0};

        double bestFMeasure = -1.0;
        String bestConfig = "";
        BayesNet bestBayesNet = null;
        int i = 1;

        // Parametro ekorketa
        for (LocalScoreSearchAlgorithm searchAlgo : bilaketaAlgoritmoak) {
            for (BayesNetEstimator estimator : estimatzaileak) {
                for (int maxParents : maxParentsValues) {

                    // SALBUESPENAK: BMAEstimator motako estimatzaileak eta K2/HillClimber ez diren algoritmoak
                    // ez dute maxParents=2 onartzen (edo ez du eraginik)
                    if (estimator instanceof BMAEstimator && maxParents > 1
                    || (!(searchAlgo instanceof K2) && !(searchAlgo instanceof HillClimber) && maxParents > 1)) {
                        break;
                    }

                    for (double alpha : alphaValues) {
                        System.out.println("Iterazioa: " + i);
                        System.out.println("Konfigurazioa probatzen: "
                                + searchAlgo.getClass().getSimpleName() + " | "
                                + estimator.getClass().getSimpleName()
                                + " | maxParents=" + maxParents
                                + " | alpha=" + alpha);
                        long hasiera = System.nanoTime();

                        BayesNet bayesNet = getBayesNet(searchAlgo, estimator, maxParents, alpha);
                        try {
                            bayesNet.buildClassifier(train);
                        } catch (Exception e) {
                            e.printStackTrace();
                            continue;
                        }

                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(bayesNet, test);

                        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
                        System.out.println("Exekuzio-denbora: " + segundoak + "s");

                        System.out.println(eval.toMatrixString());

                        // Metrikak atera
                        double fMeasure = eval.weightedFMeasure();
                        if (Double.isNaN(fMeasure)) {
                            fMeasure = 0.0;
                        }
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

        SerializationHelper.write("data/bestBayesNet.model", bestBayesNet);

        if (bestBayesNet != null) {
            String[] parametroak = bestBayesNet.getOptions();
            String parametroakTxt = Utils.joinOptions(parametroak);
            Files.write(Paths.get("data/bestBayesNetConfig.txt"), parametroakTxt.getBytes());
        } else System.out.println("Errorea: ez da eredu optimorik aurkitu.");
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

    private void txikituDatuak() throws Exception {
        System.out.println("Train instantziak hasieran: " + train.numInstances());
        StratifiedRemoveFolds srmTrain = new StratifiedRemoveFolds();
        srmTrain.setNumFolds(5);
        srmTrain.setFold(1);
        srmTrain.setInvertSelection(false);
        srmTrain.setInputFormat(train);
        train = Filter.useFilter(train, srmTrain);
        System.out.println("Train instantziak txikitu ondoren: " + train.numInstances());

        System.out.println("Test instantziak hasieran: " + test.numInstances());
        StratifiedRemoveFolds srmTest = new StratifiedRemoveFolds();
        srmTest.setNumFolds(5);
        srmTest.setFold(1);
        srmTest.setInvertSelection(false);
        srmTest.setInputFormat(test);
        test = Filter.useFilter(test, srmTest);
        System.out.println("Test instantziak txikitu ondoren: " + test.numInstances());
    }
}