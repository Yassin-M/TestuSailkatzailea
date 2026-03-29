package main;

import main.datuak.Bektorizazioa;
import main.datuak.CSV2Arff;
import main.datuak.Preprocessing;
import main.ebaluazioa.KalitateTxostena;
import main.iragarpenak.Iragarpenak;
import main.sailkatzailea.BayesNetFineTuning;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Scanner;

/**
 * Aplikazioaren klase nagusia.
 *
 * <p>Kontsolako menu baten bidez testu sailkatzailea sortzeko
 * pipeline-aren fase desberdinak exekutatzeko aukera ematen du:</p>
 *
 * <ul>
 *   <li>CSV-tik ARFF-rako bihurketa eta hasierako garbiketa</li>
 *   <li>Datuen aurreprozesamendua</li>
 *   <li>BayesNet sailkatzailearen parametroen fine-tuning</li>
 *   <li>Kalitate estimatzea eta kalitate txostena sortzea</li>
 *   <li>Iragarpenak egitea</li>
 *   <li>Pipeline osoa exekutatzea</li>
 * </ul>
 */
public class Main {

    /**
     * Kontsolatik sarrerak irakurtzeko Scanner partekatua.
     */
    private static final Scanner sc = new Scanner(System.in);

    /**
     * <p>Menua erakusten da erabiltzaileak irteteko aukera hautatu arte.</p>
     */
    public static void main(String[] ignoredArgs) {
        boolean atera = false;

        while (!atera) {
            menuaInprimatu();
            String aukera = sc.nextLine().trim();

            try {
                switch (aukera) {
                    case "1":
                        exekutatuCsvToArff();
                        break;
                    case "2":
                        exekutatuPreprocessing();
                        break;
                    case "3":
                        exekutatuBektorizazioa();
                        break;
                    case "4":
                        exekutatuDatuakBektorizatu();
                        break;
                    case "5":
                        exekutatuFineTuning();
                        break;
                    case "6":
                        exekutatuKalitateTxostena();
                        break;
                    case "7":
                        exekutatuIragarpenak();
                        break;
                    case "8":
                        exekutatuPipelineOsoa();
                        break;
                    case "0":
                        atera = true;
                        System.out.println("Agur!");
                        break;
                    default:
                        System.out.println("Aukera baliogabea. Saiatu berriro.");
                }
            } catch (Exception e) {
                System.err.println("Errorea: " + e.getMessage());
                //noinspection CallToPrintStackTrace
                e.printStackTrace();
            }

            if (!atera) {
                System.out.println();
            }
        }
    }

    /**
     * Menu nagusia kontsolan inprimatzen du.
     */
    private static void menuaInprimatu() {
        System.out.println("---- HASIERAKO MENUA ----");
        System.out.println("Aukeratu honako aukeretako bat:");
        System.out.println("1) CSV -> ARFF");
        System.out.println("2) Aurreprozesamendua (tweet-ak garbitu)");
        System.out.println("3) Bektorizazio mota optimoa aukeratu");
        System.out.println("4) Datuak bektorizatu");
        System.out.println("5) Sailkatzailearen parametro optimoak ekortu");
        System.out.println("6) Kalitatearen estimazioa egin");
        System.out.println("7) Iragarpenak egin");
        System.out.println("8) Pipeline osoa exekutatu");
        System.out.println("0) Irten");
        System.out.print("> ");
    }

    /**
     * CSV fitxategia ARFF formatura bihurtzen du (garbiketa barne).
     *
     * @throws Exception bihurketa-prozesuan errorea gertatuz gero
     */
    private static void exekutatuCsvToArff() throws Exception {
        System.out.print("Sartu input CSV bidea: ");
        String csvPath = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        CSV2Arff.arffPasatu(csvPath);
        System.out.println("CSV -> ARFF amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * ARFF fitxategi baten aurreprozesamendua exekutatzen du.
     *
     * @throws Exception datuak kargatzean edo gordetzean errorea gertatuz gero
     */
    private static void exekutatuPreprocessing() throws Exception {
        System.out.print("Sartu ARFF bidea (garbitzeko): ");
        String arffPath = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        Preprocessing.tweetakGarbitu(arffPath);
        System.out.println("Preprocessing amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Bektorizazio prozesua exekutatzen du, bektorizazio mota egokiena aukeratuz eta datuak bektorizatuz.
     *
     * @throws Exception datuak kargatzean edo gordetzean errorea gertatuz gero
     */
    private static void exekutatuBektorizazioa() throws Exception {
        System.out.print("Sartu train ARFF bidea: ");
        String train = sc.nextLine().trim();
        System.out.print("Sartu test ARFF bidea: ");
        String test = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        Instances trainInstantziak = new DataSource(train).getDataSet();
        Instances testInstantziak = new DataSource(test).getDataSet();
        Bektorizazioa.bektorizazioMotaEgokienaAztertu(trainInstantziak, testInstantziak);

        System.out.println("Bektorizazioa amaituta.");
        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    public static void exekutatuDatuakBektorizatu() throws Exception {
        System.out.print("Sartu train ARFF bidea: ");
        String trainBek = sc.nextLine().trim();
        System.out.print("Sartu dev ARFF bidea: ");
        String devBek = sc.nextLine().trim();
        System.out.print("Sartu test_blind ARFF bidea: ");
        String testBlindBek = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        Instances train = new DataSource(trainBek).getDataSet();
        Instances dev = new DataSource(devBek).getDataSet();
        Instances testBlind  = new DataSource(testBlindBek).getDataSet();
        Bektorizazioa.datuakBektorizatu(train, dev, testBlind);
        System.out.println("Datuak bektorizatu eta gorde dira.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * BayesNet sailkatzailearen parametroen fine-tuning prozesua exekutatzen du.
     *
     * @throws Exception optimizazio prozesuan errorea gertatuz gero
     */
    private static void exekutatuFineTuning() throws Exception {
        System.out.print("Sartu bektorizatutako train ARFF bidea: ");
        String trainBek = sc.nextLine().trim();
        System.out.print("Sartu bektorizatutako test ARFF bidea: ");
        String testBek = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        Instances instantziak = datuakBateratu(trainBek,testBek);
        BayesNetFineTuning.getFineTuning().fineTune(instantziak);
        System.out.println("Fine-tuning amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Ereduaren kalitate-txostena sortzen du.
     *
     * @throws Exception ebaluazioan edo fitxategia idaztean errorea gertatuz gero
     */
    private static void exekutatuKalitateTxostena() throws Exception {
        System.out.print("Sartu bektorizatutako train ARFF bidea: ");
        String trainBek = sc.nextLine().trim();
        System.out.print("Sartu bektorizatutako test ARFF bidea: ");
        String testBek = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        KalitateTxostena.kalitateaEstimatu(trainBek, testBek);
        System.out.println("Kalitate txostena sortuta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Test blind CSV baten gainean iragarpenak exekutatzen ditu.
     *
     * @throws Exception iragarpen-fasean errorea gertatuz gero
     */
    private static void exekutatuIragarpenak() throws Exception {
        System.out.print("Sartu test_blind CSV bidea: ");
        String csvPath = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        Iragarpenak.main(csvPath);
        System.out.println("Iragarpenak amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Pipeline osoa exekutatzen du.
     *
     * @throws Exception pipelineko edozein fasetan errorea gertatuz gero
     */
    private static void exekutatuPipelineOsoa() throws Exception {
        System.out.print("Sartu train datuak:");
        String csvTrain = sc.nextLine().trim();
        System.out.print("Sartu dev datuak:");
        String csvDev = sc.nextLine().trim();
        System.out.print("Sartu test_blind datuak:");
        String csvTestBlind = sc.nextLine().trim();

        // 1) CSV -> ARFF
        CSV2Arff.arffPasatu(csvTrain);
        CSV2Arff.arffPasatu(csvDev);
        CSV2Arff.arffPasatu(csvTestBlind);

        // 2) Aurreprozesamendua
        Preprocessing.tweetakGarbitu("data/arff/sortaGarbia.arff");

        // 3) Bektorizazioa
        Bektorizazioa.bektorizazioMotaEgokienaAztertu(
                new DataSource("data/arff/tweetSentiment.train.arff").getDataSet(),
                new DataSource("data/arff/tweetSentiment.dev.arff").getDataSet()
        );

        String trainBek = "data/bektorizatuak/trainBek.arff";
        String devBek = "data/bektorizatuak/devBek.arff";
        String testBlindBek = "data/bektorizatuak/testBlindBek.arff";

        // 4) Fine-tuning (bektorizatutako ARFF fitxategia behar da)
        Instances instantziak = datuakBateratu("data/arff/tweetSentiment.train.arff", "data/arff/tweetSentiment.dev.arff");
        BayesNetFineTuning.getFineTuning().fineTune(instantziak);

        // 5) Kalitate txostena
        KalitateTxostena.kalitateaEstimatu(trainBek, devBek);

        // 6) Iragarpenak
        Iragarpenak.main(testBlindBek);

        System.out.println("Pipeline osoa amaituta.");
    }

    // HELPERS
    private static Instances datuakBateratu(String datu1, String datu2) throws Exception {
        Instances instantziak1 = new DataSource(datu1).getDataSet();
        Instances instantziak2 = new DataSource(datu2).getDataSet();

        if (!instantziak1.equalHeaders(instantziak2)) {
            throw new Exception("Errorea: Datuak ez dute header berdinak.");
        }

        for (int i = 0; i < instantziak2.numInstances(); i++) {
            instantziak1.add(instantziak2.instance(i));
        }

        return instantziak1;
    }
}
