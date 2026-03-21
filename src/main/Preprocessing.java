package main;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

import java.io.File;

public class Preprocessing {

    public static void main(String[] args) throws Exception{
        String[] sortak = {"train", "dev", "test_blind"};
        for(String s: sortak){
            String path = "data/arff/tweetSentiment."+s+".arff";
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
            Instances data = source.getDataSet();

            int textIndex = data.attribute("TweetText").index();

            for(int i = 0; i<data.numInstances(); i++){
                Instance unekoa = data.instance(i);

                String tweetGarbia = cleanTweet(unekoa.stringValue(textIndex));

                unekoa.setValue(textIndex, tweetGarbia);
            }

            ArffSaver saver = new ArffSaver();
            saver.setFile(new File(path));
            saver.setInstances(data);
            saver.writeBatch();
        }
    }

    public static String cleanTweet(String tweet){
        //tweet-ak garbitu
        //url-ak ezabatu
        tweet = tweet.replaceAll("http\\S+"," ");
        //beste erabiltzaileen aipamenak @ kendu
        tweet = tweet.replaceAll("@\\w+\\s?"," ");
        //# karaktereak kendu
        tweet = tweet.replaceAll("#"," ");
        //# lotuta duten hitzak banatu
        tweet = tweet.replaceAll("([a-z])([A-Z])", "$1 $2");

        tweet = tweet.replaceAll("[^a-zA-ZáéíóúÁÉÍÓÚñÑ\\s]", "");
        tweet = tweet.replaceAll("\\s+", " ").trim();
        return tweet;
    }
}
