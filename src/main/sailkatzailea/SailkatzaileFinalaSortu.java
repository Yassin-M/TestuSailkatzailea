package main.sailkatzailea;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Emandako konfigurazio hoberena erabiliz, BayesNet sailkatzaile finala entrenatu eta
 * gordetzeaz arduratzen den klasea.
 */
public class SailkatzaileFinalaSortu {

    /**
     * Sailkatzaile finala entrenatzen du jasotako konfigurazio-parametroekin eta
     * jasotako instantzia guztiekin. Behin entrenamendua amaituta,
     * eredua data/eredua/sailkatzaileFinala.model path-era esportatzen du.
     *
     * @param configHoberena Weka formatuan dauden ereduaren parametroak dituen String-a.
     *                       Fine-tuning prozesuan automatikoki sortzen eta gordetzen da.
     * @param datuGuztiak Entrenamendurako erabiliko den datu-sorta (Instances objektua).
     */
    public static void sailkatzaileaSortu(String configHoberena, Instances datuGuztiak) {
        try {
            System.out.println();
            System.out.println("### SAILKATZAILE OSOA ENTRENATZEN ###");
            System.out.println();

            if(datuGuztiak.classIndex() == -1) datuGuztiak.setClassIndex(datuGuztiak.attribute("Sentiment").index());

            BayesNet sailkatzailea = new BayesNet();
            sailkatzailea.setOptions(weka.core.Utils.splitOptions(configHoberena));
            System.out.println("Entrenamendua hasi da...");
            sailkatzailea.buildClassifier(datuGuztiak);

            System.out.println("Entrenamendua amaitu da.");

            Files.createDirectories(Paths.get("data/eredua"));
            SerializationHelper.write("data/eredua/sailkatzaileFinala.model", sailkatzailea);
            System.out.println("Eredua data/eredua/sailkatzaileFinala.model fitxategian gorde da.");
        }
        catch (Exception e) {
            System.err.println("Errorea: " + e.getMessage());
            //noinspection CallToPrintStackTrace
            e.printStackTrace();
        }
    }
}
