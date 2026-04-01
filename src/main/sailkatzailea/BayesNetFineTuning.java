package main.sailkatzailea;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/**
 * BayesNet sailkatzailearen parametro-ekorketa prozesua kudeatzen duen klasea.
 * Estimazio-metodoak, bilaketa-algoritmoak, guraso-kopuru maximoa (maxParents) eta
 * alpha balioak optimizatzen ditu Bayes sare batentzat.
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
     * proba ebaluazio batean (5-fold cross-validation erabiliz).
     * Emaitza onena lortzen duen eredua eta bere konfigurazioa diskora gordetzen ditu.
     *
     * @param datuak Ebaluatzeko erabiliko den Weka Instances objektua (bektorizatutako datuekin).
     * @throws Exception Ebaluazioan edo eredua gordetzean arazoren bat gertatzen bada.
     */
    public void fineTune(Instances datuak) throws Exception {

        // KLASEA EZARRI
        if(datuak.classIndex() == -1) datuak.setClassIndex(datuak.attribute("Sentiment").index());

        System.out.println("BayesNet-en fine-tuning prozesua abiarazten...\n");

        // PARAMETROAK
        BayesNetEstimator[] estimatzaileak = {
                new BMAEstimator(),
                new SimpleEstimator(),
        };

        LocalScoreSearchAlgorithm[] bilaketaAlgoritmoak = {
                new HillClimber(),
                new K2(),
                new TAN()
        };

        int[] maxParentsValues = {1, 2, 3};
        double[] alphaValues = {0.01, 0.1, 0.5, 1.0, 1.5, 2.0};

        double bestFMeasure = -1.0;
        String bestConfig = "";
        BayesNet bestBayesNet = null;
        int i = 1;

        // Parametro ekorketa
        for (LocalScoreSearchAlgorithm searchAlgo : bilaketaAlgoritmoak) {
            for (BayesNetEstimator estimator : estimatzaileak) {
                if (estimator instanceof BMAEstimator && (searchAlgo instanceof TAN)) {
                    continue;
                }
                for (int maxParents : maxParentsValues) {
                    // BMAEstimator-ek bakarrik maxParents=1 balioa har dezake, eta aldi berean,
                    // maxParents bakarrik erabili datiko K2 eta HillClimber algoritmoetan.
                    // Beraz, horietarako maxParents=1 balioa bakarrik probatuko da.
                    if ((estimator instanceof BMAEstimator && maxParents > 1)
                            || (!(searchAlgo instanceof K2) && maxParents > 1)) {
                        continue;
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

                        // 5-fold Cross-Validation
                        Evaluation eval = new Evaluation(datuak);
                        try {
                            eval.crossValidateModel(bayesNet, datuak, 5, new Random(1));
                        } catch (Exception e) {
                            System.err.println("Errorea ebaluazioan: " + e.getMessage());
                            continue; // Hurrengo konfiguraziora salto egin
                        }

                        System.out.println(eval.toMatrixString());

                        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
                        System.out.println("Exekuzio-denbora: " + segundoak + "s");

                        // Metrikak atera
                        double fMeasure = eval.weightedFMeasure();
                        if (Double.isNaN(fMeasure)) {
                            fMeasure = 0.0;
                        }
                        double accuracy = eval.pctCorrect();

                        String algoName = searchAlgo.getClass().getSimpleName();
                        String estimatorName = estimator.getClass().getSimpleName();

                        System.out.printf("Emaitza -> Accuracy: %.2f%% | Weighted F-Measure: %.4f%n", accuracy, fMeasure);
                        System.out.println("---------------------------------------------------");
                        System.out.println();

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

        // Eredua gorde
        if (bestBayesNet != null) {
            Files.createDirectories(Paths.get("data/eredua"));
            SerializationHelper.write("data/eredua/bestBayesNet.model", bestBayesNet);
            String[] parametroak = bestBayesNet.getOptions();
            String parametroakTxt = Utils.joinOptions(parametroak);
            Files.write(Paths.get("data/eredua/bestBayesNetConfig.txt"), parametroakTxt.getBytes());
            System.out.println("Eredu eta konfigurazio optimoa data/eredua/ direktorioan gorde dira.");
        } else System.out.println("Errorea: ez da eredu optimorik aurkitu.");
    }

    /**
     * BayesNet sailkatzailearen instantzia bat sortzen eta konfiguratzen du emandako espezifikazioekin.
     *
     * @param searchAlgo Erabili beharreko bilaketa-algoritmoa (LocalScoreSearchAlgorithm motakoa).
     * @param estimator Erabili beharreko sare bayestarraren estimatzailea.
     * @param maxParents Guraso nodoen kopuru maximoa (algoritmoaren konplexutasuna mugatzeko K2 eta HillClimber-en).
     * @param alpha Alpha parametroa estimatzailearentzat (leuntze-parametroa).
     * @return Parametro horiekin konfiguratutako BayesNet objektua.
     */
    private BayesNet getBayesNet(LocalScoreSearchAlgorithm searchAlgo, BayesNetEstimator estimator, int maxParents, double alpha) {
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