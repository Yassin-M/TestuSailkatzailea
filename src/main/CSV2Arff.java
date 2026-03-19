package main;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class CSV2Arff {
    public static void main(String[] args) throws Exception{
        String inputPath = "/home/yassin/Descargas/Data/tweetSentiment.train.csv";
        String cleanedPath = "/home/yassin/Descargas/Data/Clean/tweetSentiment.train.csv";
        String outputPath = "/home/yassin/Descargas/Data/Clean/tweetSentiment.train.arff";
        cleanCSV(inputPath, cleanedPath);

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(cleanedPath));

        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }

    public static void cleanCSV(String input, String cleanPath){
        int esperotutakoZutabeak = 0;
        int lerroEzabatuak = 0;
        int lerroTotalak = 0;

        try(BufferedReader br = new BufferedReader(new FileReader(input, StandardCharsets.UTF_8));
            BufferedWriter bw = new BufferedWriter(new FileWriter(cleanPath, StandardCharsets.UTF_8))){
            String unekoLerroa;
            boolean lehenLerroa = true;
            while((unekoLerroa = br.readLine())!=null){
                lerroTotalak++;
                String trimmedLerroa = unekoLerroa.trim();

                if(trimmedLerroa.isEmpty()) continue;

                if(!hasBalancedQuotes(trimmedLerroa)){
                    lerroEzabatuak++;
                    continue;
                }

                if(!trimmedLerroa.endsWith("\"")){
                    lerroEzabatuak++;
                    continue;
                }

                String[] zutabeak = trimmedLerroa.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if(lehenLerroa){
                    esperotutakoZutabeak = zutabeak.length;
                    bw.write(reconstruirLinea(zutabeak));
                    bw.newLine();
                    lehenLerroa = false;
                }else{
                    if(zutabeak.length==esperotutakoZutabeak){
                        bw.write(reconstruirLinea(zutabeak));
                        bw.newLine();
                    }else{
                        lerroEzabatuak++;
                    }
                }
            }
            System.out.println("Prozesatutako lerroak: " + lerroTotalak);
            System.out.println("Ezabatutako lerroak: " + lerroEzabatuak);
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    private static String reconstruirLinea(String[] zutabeak) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < zutabeak.length; i++) {
            String gelaxka = zutabeak[i].trim();
            gelaxka = gelaxka.replace("“", "'")
                    .replace("”", "'")
                    .replace("‘", "'")
                    .replace("’", "'");

            gelaxka = gelaxka.replaceAll("[\\p{Cntrl}&&[^\r\n\t]]", "");

            if (gelaxka.startsWith("\"") && gelaxka.endsWith("\"") && gelaxka.length() >= 2) {
                gelaxka = gelaxka.substring(1, gelaxka.length() - 1);
            }

            gelaxka = gelaxka.replace("\"", "'");

            sb.append("\"").append(gelaxka).append("\"");

            if (i < zutabeak.length - 1) {
                sb.append(",");
            }
        }
        return sb.toString();
    }

    private static boolean hasBalancedQuotes(String line) {
        long count = line.chars().filter(ch -> ch == '"').count();
        return count % 2 == 0;
    }
}
