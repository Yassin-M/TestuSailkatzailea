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

/**
 * Aplikazioaren klase nagusia.
 *
 * <p>Komando-lerroko argumentuen bidez testu sailkatzailea sortzeko
 * pipeline-aren fase desberdinak exekutatzeko aukera ematen du:</p>
 *
 * <ul>
 *   <li>CSV-tik ARFF-rako bihurketa eta hasierako garbiketa</li>
 *   <li>Datuen aurreprozesamendua</li>
 *   <li>BayesNet sailkatzailearen parametroen fine-tuning</li>
 *   <li>Kalitate estimatzea eta kalitate txostena sortzea</li>
 *   <li>BayesNet sailkatzaile finala sortzea, datu gutztiekin</li>
 *   <li>Iragarpenak egitea</li>
 *   <li>Pipeline osoa exekutatzea</li>
 * </ul>
 */
public class Main {

    /**
     * Aplikazioaren sarrera puntua.
     *
     * <p>Sarrerako argumentuen arabera (args) exekutatzeko ekintza zehazten da.</p>
     *
     * @param args Komando lerroko argumentuak
     */
    public static void main(String[] args) {
        if (args.length == 0) {
            inprimatuLaguntza();
            System.exit(1);
        }

        String komandoa = args[0];

        try {
            switch (komandoa) {
                case "csv2arff":
                    if (args.length < 2) argErrorea("csv2arff <csv_path>");
                    exekutatuCsvToArff(args[1]);
                    break;
                case "aurreprozesatu":
                    if (args.length < 2) argErrorea("aurreprozesatu <arff_path>");
                    exekutatuPreprocessing(args[1]);
                    break;
                case "bektorizazioOptimoa":
                    String trainBekAuk = args.length > 1 ? args[1] : "data/arff/raw/tweetSentiment.train.arff";
                    String testBekAuk = args.length > 2 ? args[2] : "data/arff/raw/tweetSentiment.dev.arff";
                    exekutatuBektorizazioa(trainBekAuk, testBekAuk);
                    break;
                case "datuakBektorizatu":
                    String trainDat = args.length > 1 ? args[1] : "data/arff/raw/tweetSentiment.train.arff";
                    String devDat = args.length > 2 ? args[2] : "data/arff/raw/tweetSentiment.dev.arff";
                    String testBlindDat = args.length > 3 ? args[3] : "data/arff/raw/tweetSentiment.test_blind.arff";
                    exekutatuDatuakBektorizatu(trainDat, devDat, testBlindDat);
                    break;
                case "parametroEkorketa":
                    String tb = args.length > 1 ? args[1] : "data/arff/bektorizatuta/train_bektorizatua.arff";
                    exekutatuFineTuning(tb);
                    break;
                case "kalitateTxostena":
                    String kTrain = args.length > 1 ? args[1] : "data/arff/bektorizatuta/train_bektorizatua.arff";
                    String kTest = args.length > 2 ? args[2] : "data/arff/bektorizatuta/dev_bektorizatua.arff";
                    exekutatuKalitateTxostena(kTrain, kTest);
                    break;
                case "sailkatzaileaSortu":
                    String config = args.length > 1 ? args[1] : "data/eredua/bestBayesNetConfig.txt";
                    String sTrain = args.length > 2 ? args[2] : "data/arff/bektorizatuta/train_bektorizatua.arff";
                    String sTest = args.length > 3 ? args[3] : "data/arff/bektorizatuta/dev_bektorizatua.arff";
                    exekutatuSailkatzaileFinala(config, sTrain, sTest);
                    break;
                case "iragarpenakEgin":
                    String iragCsv = args.length > 1 ? args[1] : "data/tweetSentiment.test_blind.csv";
                    exekutatuIragarpenak(iragCsv);
                    break;
                case "pipelineOsoa":
                    String pTrain = args.length > 1 ? args[1] : "data/tweetSentiment.train.csv";
                    String pDev = args.length > 2 ? args[2] : "data/tweetSentiment.dev.csv";
                    String pTest = args.length > 3 ? args[3] : "data/tweetSentiment.test_blind.csv";
                    exekutatuPipelineOsoa(pTrain, pDev, pTest);
                    break;
                default:
                    System.err.println("Aukera baliogabea: " + komandoa);
                    inprimatuLaguntza();
                    System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Errorea exekuzioan: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Argumentuen erabileran errore bat dagoela inprimatu eta programa ixten du.
     *
     * @param format Erabileraren formatu zuzena argitzen duen testua
     */
    private static void argErrorea(String format) {
        System.err.println("Argumentu falta. Erabilera: " + format);
        System.exit(1);
    }

    /**
     * Programaren erabileraren argibideak eta erabili daitezkeen komandoak inprimatzen ditu.
     */
    private static void inprimatuLaguntza() {
        System.out.println("---- ERABILERA ----");
        System.out.println("java -jar TestuSailkatzailea.jar <KOMANDOA> [PARAMETROAK]");
        System.out.println("Komando onartuak:");
        System.out.println("  csv2arff <csv_path>");
        System.out.println("  aurreprozesatu <arff_path>");
        System.out.println("  bektorizazioOptimoa [train_arff] [test_arff]");
        System.out.println("  datuakBektorizatu [train_arff] [dev_arff] [test_blind_arff]");
        System.out.println("  parametroEkorketa [train_bektorizatua_arff]");
        System.out.println("  kalitateTxostena [train_bektorizatua_arff] [test_bektorizatua_arff]");
        System.out.println("  sailkatzaileaSortu [config_txt] [train_bektorizatua_arff] [test_bektorizatua_arff]");
        System.out.println("  iragarpenakEgin [test_blind_csv]");
        System.out.println("  pipelineOsoa [train_csv] [dev_csv] [test_blind_csv]");
    }

    /**
     * CSV fitxategi bat ARFF formatura bihurtzeko prozesua exekutatzen du.
     *
     * @param csvPath Irakurri nahi den CSV fitxategiaren path
     * @throws Exception Fitxategia irakurtzean edo bihurtzean arazoren bat badago
     */
    private static void exekutatuCsvToArff(String csvPath) throws Exception {
        long hasiera = System.nanoTime();

        CSV2Arff.arffPasatu(csvPath);
        System.out.println();
        System.out.println("CSV -> ARFF amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * ARFF fitxategiko datuen aurreprozesamendua eta garbiketa exekutatzen du.
     *
     * @param arffPath Garbitu nahi den ARFF fitxategiaren path-a
     * @throws Exception Garbiketako erroreak kudeatzeko
     */
    private static void exekutatuPreprocessing(String arffPath) throws Exception {
        long hasiera = System.nanoTime();

        Preprocessing.tweetakGarbitu(arffPath);
        System.out.println();
        System.out.println("Aurreprozesamendua amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Datuentzat bektorizazio mota egokiena bilatzen du prozesuko ezaugarri onenak ateraz.
     *
     * @param train Entrenamendu multzoaren ARFF path-a
     * @param test Proba multzoaren ARFF path-a
     * @throws Exception Datuak irakurtzean edo bilaketa egitean errorea bada
     */
    private static void exekutatuBektorizazioa(String train, String test) throws Exception {
        long hasiera = System.nanoTime();

        Instances trainInstantziak = new DataSource(train).getDataSet();
        Instances testInstantziak = new DataSource(test).getDataSet();
        Bektorizazioa.bektorizazioMotaEgokienaAztertu(trainInstantziak, testInstantziak, 700);

        System.out.println();
        System.out.println("Bektorizazioa amaituta.");
        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Dauden datu multzo guztiak bektorizatzen ditu ezaugarriak erabilka.
     *
     * @param trainBek Entrenamendu datuen jatorrizko ARFF fitxategiaren path-a
     * @param devBek Garapen (dev) datuen jatorrizko ARFF fitxategiaren path-a
     * @param testBlindBek Test itsu (blind) datuen jatorrizko ARFF fitxategiaren path-a
     * @throws Exception Datuak irakurtzean edo iragazkiak aplikatzean errorea ematen badu
     */
    private static void exekutatuDatuakBektorizatu(String trainBek, String devBek, String testBlindBek) throws Exception {
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
     * BayesNet sailkatzailearen parametro ekorketa (Fine-Tuning) burutzen du aukera onena lortzeko.
     *
     * @param trainBek Entrenamendurako prestatuta dagoen fitxategi bektorizatuaren path-a
     * @throws Exception Ikasketa fasean optimizazioak akatsen bat badu
     */
    private static void exekutatuFineTuning(String trainBek) throws Exception {
        long hasiera = System.nanoTime();

        Instances datuak = new DataSource(trainBek).getDataSet();
        BayesNetFineTuning.getFineTuning().fineTune(datuak);

        System.out.println();
        System.out.println("Fine-tuning amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Sailkatzailearen errendimendua neurtu eta kalitate txosten oso bat gordetzen du.
     *
     * @param trainBek Entrenamendurako datu bektorizatuen path-a
     * @param testBek Probarako (edo dev) datu bektorizatuen path-a
     * @throws Exception Ebaluazio prozesuan zehar gertatutako erroreak kudeatzeko
     */
    private static void exekutatuKalitateTxostena(String trainBek, String testBek) throws Exception {
        long hasiera = System.nanoTime();

        KalitateTxostena.kalitateaEstimatu(trainBek, testBek);

        System.out.println();
        System.out.println("Kalitate txostena sortuta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Datu guztiak uztartu eta lortutako parametro optimoenekin amaierako sailkatzailea sortzen du.
     *
     * @param config Parametroen ezarpenak dituen testu fitxategiaren path-a
     * @param trainBek Entrenamendu datuen multzo partziala
     * @param testBek Proba (dev) datuen multzo partziala
     * @throws Exception Sailkatzailea azken datuekin entrenatzean edo fitxategian gordetzean
     */
    private static void exekutatuSailkatzaileFinala(String config, String trainBek, String testBek) throws Exception {
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
     * Test itsuko datuekin sailkapen berriak egiten ditu lehendik dagoen Eredua erabilita.
     *
     * @param csvPath Test itsuko datuen hasierako CSV path-a
     * @throws Exception Datuak irakurtzean edo aurreikuspenak prozesatzean
     */
    private static void exekutatuIragarpenak(String csvPath) throws Exception {
        long hasiera = System.nanoTime();

        Iragarpenak.iragarpenakEgin(csvPath);

        System.out.println();
        System.out.println("Iragarpenak amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.println("Exekuzio-denbora: " + segundoak + "s");
    }

    /**
     * Finkatutako prozesu osoa automatikoki aurrera eramaten du, hasieratik bukaerara (urratsez urrats).
     *
     * @param csvTrain Entrenamendurako bideratutako hasierako CSV fitxategia
     * @param csvDev Garapenerako bideratutako CSV fitxategia
     * @param csvTestBlind Test itsua (amaierako baieztapena) duen CSV fitxategia
     * @throws Exception Edozein ataletan akats larria ematen bada
     */
    private static void exekutatuPipelineOsoa(String csvTrain, String csvDev, String csvTestBlind) throws Exception {
        System.out.println();
        System.out.println(" ### PIPELINE OSOA EXEKUTATZEN ### ");
        System.out.println();


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
        System.out.println("\n--- 3. Bektorizazio mota optimoa aztertu ---");
        System.out.println();
        Bektorizazioa.bektorizazioMotaEgokienaAztertu(trainInstantziak, devInstantziak, 700);

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

    /**
     * Bi ARFF fitxategien datuak batean fusionatzen ditu.
     *
     * @param datu1 Lehenengo ARFF fitxategiaren path-a
     * @param datu2 Bigarren ARFF fitxategiaren path-a
     * @return Bi datuen multzoen batuketa, lehenengoak deskribaturiko estruktura mantenduta
     * @throws Exception fitxategiak kargatzean, header-aren parekatzean edo header desberdinak direnean errorea gertatuz gero
     */
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
