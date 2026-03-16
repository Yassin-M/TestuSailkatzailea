package main;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class CSV2Arff {
    public static void main(String[] args) throws Exception{
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(args[1]));
        saver.setInstances(data);
        saver.writeBatch();
    }
}
