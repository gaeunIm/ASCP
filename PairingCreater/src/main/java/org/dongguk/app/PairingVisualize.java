package org.dongguk.app;

import org.dongguk.domain.Flight;
import org.dongguk.domain.Pairing;

import java.io.*;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Comparator;
import java.util.List;

import static java.lang.String.*;

public class PairingVisualize{

    List<Pairing> pairingList;
    public PairingVisualize(List<Pairing> pairingList){
        this.pairingList = pairingList;
    }

    public void visualize() {
        //첫 항공기의 출발시간을 기준으로 정렬
        pairingList.removeIf(pairing -> pairing.getPair().isEmpty());
        pairingList.sort(Comparator.comparing(a -> a.getPair().get(0).getOriginTime()));

        //첫 항공기의 출발시간~마지막 항공기의 도착 시간까지 타임 테이블 생성
        StringBuilder text = new StringBuilder();
        LocalDateTime f = pairingList.get(0).getPair().get(0).getOriginTime();
        LocalDateTime firstTime = stripMinutes(f);
        LocalDateTime l = firstTime;
        for (Pairing pairing : pairingList) {
            System.out.print("Set" + pairingList.indexOf(pairing) + " ] ");
            for(Flight flight : pairing.getPair()){
                if(flight.getDestTime().isAfter(l)) l = flight.getDestTime();
                System.out.print(" / " + flight.getOriginTime() +" ~ "+ flight.getDestTime());
            }
            System.out.println();
        }
        LocalDateTime lastTime = stripMinutes(l);

        //첫 줄에 날짜 단위 입력
        f = firstTime;
        text.append(",").append(f).append(",");
        f = f.plusHours(1);
        do {
            if(f.getHour()==0) text.append(f);
            text.append(",");

            f = f.plusHours(1);
        } while (!f.equals(lastTime));
        text.append("\n");

        //두번째 줄에 시간 단위 입력
        text.append(",");
        f = firstTime;
        do {
            text.append(f.getHour());
            text.append(":00,");

            f = f.plusHours(1);
        } while (!f.equals(lastTime));
        text.append("\n");

        //타임 테이블의 내용 작성
        for(Pairing pairing : pairingList){
            text.append("SET");
            text.append(pairingList.indexOf(pairing));
            text.append(",");
            String s = buildTable(pairing.getPair(), firstTime);
            text.append(s);
        }

        //csv 파일로 출력
        try (FileWriter fw = new FileWriter("visualized-data.csv")) {
            fw.write(text.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    //출발시간과 도착 시간의 차이를 구하며 csv format 에 맞는 text 생성.
    public String buildTable(List<Flight> pairing, LocalDateTime firstTime){
        StringBuilder sb = new StringBuilder();
        for(Flight flight : pairing){
            int a = (int) ChronoUnit.HOURS.between(firstTime, stripMinutes(flight.getOriginTime()));
            sb.append(",".repeat(Math.max(0,a)));
            sb.append(flight.getOriginAirport().getName());
            sb.append(",");
            int b = (int) ChronoUnit.HOURS.between(flight.getOriginTime(), stripMinutes(flight.getDestTime()));
            sb.append("#######,".repeat(Math.max(0,b-1)));
            sb.append(flight.getDestAirport().getName());
            sb.append(",");

            firstTime = stripMinutes(flight.getDestTime());
        }
        sb.append("\n");

        return valueOf(sb);
    }

    //분 단위를 버림함
    public LocalDateTime stripMinutes(LocalDateTime l){
        return LocalDateTime.of(l.getYear(), l.getMonth(), l.getDayOfMonth(), l.getHour(), 0);
    }
}