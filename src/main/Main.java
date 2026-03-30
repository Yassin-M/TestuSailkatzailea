package main;

import main.datuak.Bektorizazioa;
import main.datuak.CSV2Arff;
import main.datuak.Preprocessing;
import main.ebaluazioa.KalitateTxostena;
import main.iragarpenak.Iragarpenak;
import main.sailkatzailea.BayesNetFineTuning;
import main.sailkatzailea.SailkatzaileFinalaSortu;
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
                        exekutatuSailkatzaileFinala();
                        break;
                    case "8":
                        exekutatuIragarpenak();
                        break;
                    case "9":
                        exekutatuPipelineOsoa();
                        break;
                    case "0":
                        atera = true;
                        System.out.println("Agur!");
                        break;
                    default:
                        System.out.println("Aukera baliogabea. Saiatu berriro.");
                }
                if (!atera) {
                    System.out.println();
                    System.out.println("------ AMAITUTA ------");
                    System.out.println("Sakatu edozein tekla menura bueltatzeko...");
                    sc.nextLine();
                    System.out.println();
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
        System.out.println();
        System.out.println("---- HASIERAKO MENUA ----");
        System.out.println("Aukeratu honako aukeretako bat:");
        System.out.println("1) CSV -> ARFF");
        System.out.println("2) Aurreprozesamendua (tweet-ak garbitu)");
        System.out.println("3) Bektorizazio mota optimoa aukeratu");
        System.out.println("4) Datuak bektorizatu");
        System.out.println("5) Sailkatzailearen parametro optimoak ekortu");
        System.out.println("6) Kalitatearen estimazioa egin");
        System.out.println("7) Amaierako eredua sortu");
        System.out.println("8) Iragarpenak egin");
        System.out.println("9) Pipeline osoa exekutatu");
        System.out.println("0) Irten");
        System.out.print("> ");
    }

    /**
     * CSV fitxategia ARFF formatura bihurtzen du (garbiketa barne).
     *
     * @throws Exception bihurketa-prozesuan errorea gertatuz gero
     */
    private static void exekutatuCsvToArff() throws Exception {
        System.out.print("Sartu input CSV fitxategia: ");
        String csvPath = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        CSV2Arff.arffPasatu(csvPath);
        System.out.println();
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
        System.out.print("Sartu ARFF fitxategia (garbitzeko): ");
        String arffPath = sc.nextLine().trim();

        long hasiera = System.nanoTime();

        Preprocessing.tweetakGarbitu(arffPath);
        System.out.println();
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
        System.out.print("Sartu train ARFF fitxategia: ");
        String train = sc.nextLine().trim();
        if (train.isEmpty()) train = "data/arff/tweetSentiment.train.arff";
        System.out.print("Sartu test ARFF fitxategia: ");
        String test = sc.nextLine().trim();
        if (test.isEmpty()) test = "data/arff/tweetSentiment.dev.arff";

        long hasiera = System.nanoTime();

        Instances trainInstantziak = new DataSource(train).getDataSet();
        Instances testInstantziak = new DataSource(test).getDataSet();
        Bektorizazioa.bektorizazioMotaEgokienaAztertu(trainInstantziak, testInstantziak, 500);

        System.out.println();
        System.out.println("Bektorizazioa amaituta.");
        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    private static void exekutatuDatuakBektorizatu() throws Exception {
        System.out.print("Sartu train ARFF fitxategia: ");
        String trainBek = sc.nextLine().trim();
        if (trainBek.isEmpty()) trainBek = "data/arff/tweetSentiment.train.arff";
        System.out.print("Sartu dev ARFF fitxategia: ");
        String devBek = sc.nextLine().trim();
        if (devBek.isEmpty()) devBek = "data/arff/tweetSentiment.dev.arff";
        System.out.print("Sartu test_blind ARFF fitxategia: ");
        String testBlindBek = sc.nextLine().trim();
        if (testBlindBek.isEmpty()) testBlindBek = "data/arff/tweetSentiment.test_blind.arff";

        long hasiera = System.nanoTime();

        Instances train = new DataSource(trainBek).getDataSet();
        Instances dev = new DataSource(devBek).getDataSet();
        Instances testBlind  = new DataSource(testBlindBek).getDataSet();
        Bektorizazioa.datuakBektorizatu(train, dev, testBlind);

        System.out.println();
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
        System.out.print("Sartu bektorizatutako train ARFF fitxategia: ");
        String trainBek = sc.nextLine().trim();
        if (trainBek.isEmpty()) trainBek = "data/arff/bektorizatuta/train_bektorizatua.arff";

        long hasiera = System.nanoTime();

        Instances datuak = new DataSource(trainBek).getDataSet();
        BayesNetFineTuning.getFineTuning().fineTune(datuak);

        System.out.println();
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
        System.out.print("Sartu bektorizatutako train ARFF fitxategia: ");
        String trainBek = sc.nextLine().trim();
        if (trainBek.isEmpty()) trainBek = "data/arff/bektorizatuta/train_bektorizatua.arff";
        System.out.print("Sartu bektorizatutako test ARFF fitxategia: ");
        String testBek = sc.nextLine().trim();
        if (testBek.isEmpty()) testBek = "data/arff/bektorizatuta/dev_bektorizatua.arff";

        long hasiera = System.nanoTime();

        KalitateTxostena.kalitateaEstimatu(trainBek, testBek);

        System.out.println();
        System.out.println("Kalitate txostena sortuta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    private static void exekutatuSailkatzaileFinala() throws Exception {
        System.out.print("Sartu sailkatzaile onenaren konfigurazioaren fitxategia: ");
        String config = sc.nextLine().trim();
        if (config.isEmpty()) config = "data/eredua/bestBayesNetConfig.txt";
        System.out.print("Sartu bektorizatutako train ARFF fitxategia: ");
        String trainBek = sc.nextLine().trim();
        if (trainBek.isEmpty()) trainBek = "data/arff/bektorizatuta/train_bektorizatua.arff";
        System.out.print("Sartu bektorizatutako test ARFF fitxategia: ");
        String testBek = sc.nextLine().trim();
        if (testBek.isEmpty()) testBek = "data/arff/bektorizatuta/dev_bektorizatua.arff";

        long hasiera = System.nanoTime();

        String configString = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(config)));
        Instances datuak = datuakBateratu(trainBek, testBek);
        SailkatzaileFinalaSortu.sailkatzaileaSortu(configString, datuak);

        System.out.println();
        System.out.println("Eredu finala sortuta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Test blind CSV baten gainean iragarpenak exekutatzen ditu.
     *
     * @throws Exception iragarpen-fasean errorea gertatuz gero
     */
    private static void exekutatuIragarpenak() throws Exception {
        System.out.print("Sartu testerako CSV fitxategia: ");
        String csvPath = sc.nextLine().trim();
        if (csvPath.isEmpty()) csvPath = "tweetSentiment.test_blind.csv";

        long hasiera = System.nanoTime();

        Iragarpenak.iragarpenakEgin(csvPath);

        System.out.println();
        System.out.println("Iragarpenak amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Pipeline osoa exekutatzen du modu autonomoan.
     *
     * @throws Exception pipelineko edozein fasetan errorea gertatuz gero
     */
    private static void exekutatuPipelineOsoa() throws Exception {
        System.out.println();
        System.out.println(" ### PIPELINE OSOA EXEKUTATZEN ### ");
        System.out.println();

        System.out.print("Sartu train CSV datuak: ");
        String csvTrain = sc.nextLine().trim();
        if (csvTrain.isEmpty()) csvTrain = "data/tweetSentiment.train.csv";

        System.out.print("Sartu dev CSV datuak: ");
        String csvDev = sc.nextLine().trim();
        if (csvDev.isEmpty()) csvDev = "data/tweetSentiment.dev.csv";

        System.out.print("Sartu test_blind CSV datuak: ");
        String csvTestBlind = sc.nextLine().trim();
        if (csvTestBlind.isEmpty()) csvTestBlind = "data/tweetSentiment.test_blind.csv";

        long hasiera = System.nanoTime();

        // 1) CSV -> ARFF
        System.out.println();
        System.out.println("\n--- 1. CSV -> ARFF ---");
        System.out.println();
        CSV2Arff.arffPasatu(csvTrain);
        CSV2Arff.arffPasatu(csvDev);
        CSV2Arff.arffPasatu(csvTestBlind);

        String trainArff = "data/arff/raw/tweetSentiment.train.arff";
        String devArff = "data/arff/raw/tweetSentiment.dev.arff";
        String testBlindArff = "data/arff/raw/tweetSentiment.test_blind.arff";

        // 2) Aurreprozesamendua (Garbitu ARFF fitxategiak)
        System.out.println();
        System.out.println("\n--- 2. Aurreprozesamendua ---");
        System.out.println();
        Preprocessing.tweetakGarbitu(trainArff);
        Preprocessing.tweetakGarbitu(devArff);
        Preprocessing.tweetakGarbitu(testBlindArff);

        // Instantziak kargatu
        Instances trainInstantziak = new DataSource(trainArff).getDataSet();
        Instances devInstantziak = new DataSource(devArff).getDataSet();
        Instances testBlindInstantziak = new DataSource(testBlindArff).getDataSet();

        // 3) Bektorizazio mota optimoa aztertu
        System.out.println();
        System.out.println("\n--- 3. Bektorizazio mota optimoa aukeratu ---");
        System.out.println();
        Bektorizazioa.bektorizazioMotaEgokienaAztertu(trainInstantziak, devInstantziak, 500);

        // 4) Datuak bektorizatu
        System.out.println();
        System.out.println("\n--- 4. Datuak bektorizatu ---");
        System.out.println();
        Bektorizazioa.datuakBektorizatu(trainInstantziak, devInstantziak, testBlindInstantziak);

        String trainBek = "data/arff/bektorizatuta/train_bektorizatua.arff";
        String devBek = "data/arff/bektorizatuta/dev_bektorizatua.arff";

        // 5) Fine-tuning
        System.out.println();
        System.out.println("\n--- 5. Fine-Tuning ---");
        System.out.println();
        Instances trainBekInstantziak = new DataSource(trainBek).getDataSet();
        BayesNetFineTuning.getFineTuning().fineTune(trainBekInstantziak);

        // 6) Kalitate txostena
        System.out.println();
        System.out.println("\n--- 6. Kalitate txostena ---");
        System.out.println();
        KalitateTxostena.kalitateaEstimatu(trainBek, devBek);

        // 7) Sailkatzaile finala sortu
        System.out.println();
        System.out.println("\n--- 7. Amaierako eredua sortu ---");
        System.out.println();
        String configPath = "data/eredua/bestBayesNetConfig.txt";
        String configString = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(configPath)));
        Instances datuakBateratu = datuakBateratu(trainBek, devBek);
        SailkatzaileFinalaSortu.sailkatzaileaSortu(configString, datuakBateratu);

        // 8) Iragarpenak test_blind.csv erabiliz
        System.out.println();
        System.out.println("\n--- 8. Iragarpenak egin ---");
        System.out.println();
        Iragarpenak.iragarpenakEgin(csvTestBlind);

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("\nPipeline osoa amaituta " + segundoak + " segundotan.");
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
