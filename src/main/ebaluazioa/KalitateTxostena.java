package main.ebaluazioa;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Testu-sailkatzailearen kalitate-txostena sortzen duen klasea.
 * <p>
 * Klase honek bektorizatutako entrenamendu eta proba datuak kargatzen ditu.
 * Ondoren, aldez aurretik gordetako konfigurazio optimoa erabiliz BayesNet eredua entrenatzen du,
 * eta Hold-Out ebaluazio bat burutzen du datu ezezagunen gainean. Azkenik, lortutako metrikak
 * (nahasmen-matrizea, precision, recall, f-score, etab.) 'irteera/kalitateTxostena.txt' fitxategian gordetzen ditu.
 * </p>
 */
public class KalitateTxostena {
    /**
     * Kalitate txostenaren sorrera prozesua exekutatzen du.
     * <p>
     * Prozesu honetan honako urratsak jarraitzen dira: datu bektorizatuak kargatu,
     * klase-indizeak ezarri, ereduaren konfigurazio optimoa defektuzko testu fitxategitik irakurri,
     * BayesNet sailkatzailea entrenatu (train datuekin) eta jarraian test/dev datuekin ebaluatu.
     * Emaitza guztiak txosten batean idazten dira.
     * </p>
     *
     * @param trainBek Entrenamendurako erabiliko den .arff fitxategi bektorizatuaren path-a.
     * @param testBek Ebaluaziorako erabiliko den .arff fitxategi bektorizatuaren path-a.
     * @throws Exception Datu-fitxategiak edo konfigurazioa irakurtzean, eredua entrenatzean,
     *                   ebaluazioa egitean edo txostena idaztean gertatzen den edozein errore.
     */
    public static void kalitateaEstimatu(String trainBek, String testBek) throws Exception {
        System.out.println("Ebaluaziorako datuak kargatzen...");

        //Test-aren importazioa
        DataSource sourceTestBektorizatua = new DataSource(testBek);
        Instances testBektorizatua = sourceTestBektorizatua.getDataSet();
        DataSource sourceTrainBektorizatua = new DataSource(trainBek);
        Instances trainBektorizatua = sourceTrainBektorizatua.getDataSet();

        if (testBektorizatua.classIndex() == -1) {
            testBektorizatua.setClassIndex(testBektorizatua.numAttributes() - 1);
        }
        if (trainBektorizatua.classIndex() == -1) {
            trainBektorizatua.setClassIndex(trainBektorizatua.numAttributes() - 1);
        }

        // Konfigurazio hoberena hartu
        String configuracionTxt = new String(Files.readAllBytes(Paths.get("data/eredua/bestBayesNetConfig.txt")));

        // Eredua entrenatu eta konfigurazio hoberena pasatu
        System.out.println("Sailkatzailea entrenatzen...");
        BayesNet sailkatzailea = new BayesNet();
        sailkatzailea.setOptions(Utils.splitOptions(configuracionTxt));
        sailkatzailea.buildClassifier(trainBektorizatua);
        System.out.println("Sailkatzailea entrenatu da.");

        // Ebaluazioa egin (Hold-Out)
        System.out.println("Ebaluazioa egiten...");
        Evaluation eval = new Evaluation(trainBektorizatua);
        eval.evaluateModel(sailkatzailea, testBektorizatua);

        // Txostenaren path-a ezarri
        String resultPath = "irteera/kalitateTxostena.txt";
        Files.createDirectories(Paths.get("irteera"));

        // Ebaluzaio txostena izango duen parametroak hautatu (dokumentazioan azalpen konpletoago bat)
        try (FileWriter fw = new FileWriter(resultPath)) {
            fw.write("=== KALITATE TXOSTENA ===\n\n");
            fw.write(eval.toMatrixString("Nahasmen-matrizea:") + "\n\n");

            // Klase balio bakoitzeko datuak
            fw.write("=== Klase bakoitzarekiko ebaluazio datuak ===\n");
            for (int i = 0; i < testBektorizatua.classAttribute().numValues(); i++) {
                String klasea = testBektorizatua.classAttribute().value(i);
                fw.write("Klasea "+ klasea +":\n");
                fw.write("  - Precision: " + eval.precision(i) + "\n");
                fw.write("  - Recall:    " + eval.recall(i) + "\n");
                fw.write("  - F-Score:   " + eval.fMeasure(i) + "\n");
                fw.write("\n");
            }

            // Sailkatzaile globalaren ebaluazioaren datuak (aurreko atalean aurkeztutako metrika berdinak)
            fw.write("\n=== Sailkatzaile globalaren datuak ===\n");
            fw.write("Sailkatzailearen Accuracy: " + eval.pctCorrect() + "\n");
            fw.write("F-Score-ren batazbesteko haztatua: " + eval.weightedFMeasure() + "\n");

            // Sailkatzailearen informazio gehigarria
            fw.write("\n=== Sailkatzailearen informazio gehigarria ===\n");
            fw.write("Erabilitako sailkatzaile mota: Bayes Network (BayesNet)\n");
            fw.write("Parametro optimoak: \n" + Utils.joinOptions(sailkatzailea.getOptions()) + "\n\n");
            String bektorizazioOptions;
            try {
                bektorizazioOptions = new String(Files.readAllBytes(Paths.get(
                        "data/arff/bektorizatuta/txt/bektorizazioHoberena.txt"))).trim();
            } catch (Exception e) {
                bektorizazioOptions = "Bektorizazio fitxategia ez da aurkitu.";
            }
            fw.write("Bektorizazioaren konfigurazioa: \n" + bektorizazioOptions + "\n\n");
            fw.write("Erabili den ebaluazio eskema: Hold-Out (Train / Dev)\n");

            System.out.println("Kalitate txostena ondo gorde da: " + resultPath);

            // (FileWriterrek automatikoki egingo du flush eta close (try-with-resources))
        }
    }
}