package main.sailkatzailea;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.nio.file.Files;
import java.nio.file.Paths;

public class SailkatzaileFinalaSortu {
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
