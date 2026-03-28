package main;

import main.sailkatzailea.BayesNetFineTuning;
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
    static void main() {
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
                        exekutatuFineTuning();
                        break;
                    case "4":
                        exekutatuKalitateTxostena();
                        break;
                    case "5":
                        exekutatuIragarpenak();
                        break;
                    case "6":
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
        System.out.println("1) CSV -> ARFF (+ cleanCSV)");
        System.out.println("2) Preprocessing (tweetakGarbitu)");
        System.out.println("3) BayesNet Fine-Tuning");
        System.out.println("4) Kalitate Txostena");
        System.out.println("5) Iragarpenak");
        System.out.println("6) Pipeline osoa");
        System.out.println("0) Irten");
        System.out.print("> ");
    }

    /**
     * CSV fitxategia ARFF formatura bihurtzen du (garbiketa barne).
     *
     * @throws Exception bihurketa-prozesuan errorea gertatuz gero
     */
    private static void exekutatuCsvToArff() throws Exception {
        long hasiera = System.nanoTime();

        System.out.print("Sartu input CSV bidea: ");
        String csvPath = sc.nextLine().trim();
        CSV2Arff.arffPasatu(csvPath);
        System.out.println("CSV -> ARFF amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.printf("%nExekuzio-denbora: %.3f s%n", segundoak);
    }

    /**
     * ARFF fitxategi baten aurreprozesamendua exekutatzen du.
     *
     * @throws Exception datuak kargatzean edo gordetzean errorea gertatuz gero
     */
    private static void exekutatuPreprocessing() throws Exception {
        long hasiera = System.nanoTime();

        System.out.print("Sartu ARFF bidea (garbitzeko): ");
        String arffPath = sc.nextLine().trim();
        Preprocessing.tweetakGarbitu(arffPath);
        System.out.println("Preprocessing amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.printf("%nExekuzio-denbora: %.3f s%n", segundoak);
    }

    /**
     * BayesNet sailkatzailearen parametroen fine-tuning prozesua exekutatzen du.
     *
     * @throws Exception optimizazio prozesuan errorea gertatuz gero
     */
    private static void exekutatuFineTuning() throws Exception {
        long hasiera = System.nanoTime();

        System.out.print("Sartu bektorizatutako ARFF bidea: ");
        String vecArffPath = sc.nextLine().trim();
        DataSource ds = new DataSource(vecArffPath);
        BayesNetFineTuning.getFineTuning().fineTune(ds);
        System.out.println("Fine-tuning amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.printf("%nExekuzio-denbora: %.3f s%n", segundoak);
    }

    /**
     * Ereduaren kalitate-txostena sortzen du.
     *
     * @throws Exception ebaluazioan edo fitxategia idaztean errorea gertatuz gero
     */
    private static void exekutatuKalitateTxostena() throws Exception {
        long hasiera = System.nanoTime();

        KalitateTxostena.main();
        System.out.println("Kalitate txostena sortuta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.printf("%nExekuzio-denbora: %.3f s%n", segundoak);
    }

    /**
     * Test blind CSV baten gainean iragarpenak exekutatzen ditu.
     *
     * @throws Exception iragarpen-fasean errorea gertatuz gero
     */
    private static void exekutatuIragarpenak() throws Exception {
        long hasiera = System.nanoTime();

        System.out.print("Sartu test_blind CSV bidea: ");
        String csvPath = sc.nextLine().trim();
        Iragarpenak.main(csvPath);
        System.out.println("Iragarpenak amaituta.");

        double segundoak = (System.nanoTime() - hasiera) / 1_000_000_000.0;
        System.out.printf("%nExekuzio-denbora: %.3f s%n", segundoak);
    }

    /**
     * Pipeline osoa exekutatzen du.
     *
     * @throws Exception pipelineko edozein fasetan errorea gertatuz gero
     */
    private static void exekutatuPipelineOsoa() throws Exception {
        System.out.print("Sartu train datuak:");
        String csvPath = sc.nextLine().trim();

        // 1) CSV -> ARFF
        CSV2Arff.arffPasatu(csvPath);

        // 2) Aurreprozesamendua
        Preprocessing.tweetakGarbitu("data/arff/sortaGarbia.arff");

        // 3) Fine-tuning (bektorizatutako ARFF fitxategia behar da)
        BayesNetFineTuning.getFineTuning().fineTune(new DataSource("data/arff/train_vectorized.arff"));

        // 4) Kalitate txostena
        KalitateTxostena.main();

        // 5) Iragarpenak
        Iragarpenak.main(csvPath);

        System.out.println("Pipeline osoa amaituta.");
    }
}
