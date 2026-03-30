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
 * Testu-sailkatzailearen kalitate-txostena sortzen duen utilitate klasea.
 * <p>
 * Klase honek bektorizatutako entrenamendu eta proba datuak kargatzen ditu,
 * BayesNet eredua konfigurazio onenarekin berreraiki (lehenik sailkatzaile hutsa kargatzen du)
 * eta entrenatzen du, eta sortzen duen ebaluazio eskema (kalitateTxostena.txt) fitxategian gorde.
 * </p>
 */
public class KalitateTxostena {
    /**
     * Kalitate txostenaren sorrera exekutatzen du.
     * <p>
     * Exekuzioan urrats hauek egiten dira:
     * entrenamendu/proba datuak kargatu, klase-indizea ezarri,
     * BayesNet eredu hutsa kargatu, konfigurazio-fitxategia irakurri, BayesNet eredua entrenatu,
     * ebaluazioa egin eta azken txostena fitxategian idatzi.
     * </p>
     *
     * @throws Exception Datu-fitxategiak, eredua edo konfigurazioa irakurri/erabiltzean
     *                   edo txostena idaztean gertatzen den edozein errore.
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