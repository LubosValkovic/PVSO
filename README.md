 Zadanie 2 – Kalibrácia kamery a spracovanie obrazu

 Popis
Projekt je zameraný na:
- kalibráciu kamery Ximea pomocou šachovnice,
- spracovanie obrazu v knižnici OpenCV,
- detekciu kruhov pomocou Houghovej transformácie,
- detekciu základných geometrických tvarov,
- prácu s farebnými maskami a filtráciou farieb v reálnom čase.

 Cieľ
Cieľom zadania je získať kalibračné parametre kamery, použiť ich pri spracovaní obrazu a následne v obraze rozpoznávať vybrané tvary a farby.

 Použité technológie
- Python
- OpenCV
- NumPy
- Ximea kamera

 Hlavné časti projektu

 1. Kalibrácia kamery
Program využíva obrázky šachovnice na:
- nájdenie rohov šachovnice,
- spresnenie ich polohy,
- výpočet matice kamery a skreslenia,
- uloženie kalibračných parametrov.

 2. Detekcia kruhov
Na detekciu kruhov sa používa:
- prevod obrazu do grayscale,
- filtrovanie obrazu,
- Houghova transformácia.

 3. Detekcia geometrických tvarov
Program vyhľadáva kontúry objektov a podľa počtu vrcholov rozlišuje:
- trojuholník,
- štvorec,
- obdĺžnik.

 4. Výpočet stredu objektu
Pri nájdených objektoch sa počíta stred pomocou obrazových momentov.

 5. Farebná maska
Program pracuje s HSV priestorom a maskami na detekciu farieb. Pomocou sliderov je možné meniť výslednú farbu zvýraznenia.

 Výstup
Výsledkom je:
- kalibrovaný obraz kamery,
- detekcia tvarov v obraze,
- zvýraznenie vybraných farebných oblastí.
