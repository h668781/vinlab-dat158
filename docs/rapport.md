# Vinlab vinkvalitet med maskinlæring

DAT158 · 23. oktober 2026 · Namn: Stian Myrene Holstad

## 1 Beskriv problemet

Vinlab er ei nettside som prøver å føreseie vinkvalitet ut frå kjemiske målingar. Brukaren fyller inn elleve verdiar og får eit estimat frå 0 til 10. Prosjektet er ein enkel demo av maskinlæring. Det er ikkje meint å erstatte eit smakspanel eller seie om ein vin er trygg å drikke.

Dette er eit regresjonsproblem. Hovudmålet er låg MAE, altså gjennomsnittleg absolutt feil i kvalitetspoeng. RMSE og R² blir også målte. Ein modell som alltid svarar medianen, er med som baseline. Han viser om dei andre modellane faktisk tilfører noko.

## 2 Data

Dataa er henta frå UCI Wine Quality [1]. Prosjektet bruker berre raudvin, med 1599 rader. Input er mellom anna alkohol, syre, sukker og pH. Målet quality er basert på ekspertvurderingar. Dei fleste vinane har fått 5 eller 6 poeng, så det er lite data om dei beste og dårlegaste vinane.

Det er ingen manglande verdiar. Like målerader blei fjerna før oppdeling, med den første vurderinga bevart. Då stod det att 1359 rader. Med seed 158 blei dei delte i 815 til trening, 272 til validering og 272 til test. Testdataa blei haldne utanfor modellvalet. Ridge bruker skalering berekna berre frå treningsdataa. Trærne treng ikkje slik skalering.

## 3 Modellering

Median, ridge-regresjon, beslutningstre og random forest blei samanlikna. Modellane er skrivne i Python med NumPy. Eit tre deler dataa ved tersklar som reduserer kvadratisk feil. Ein random forest bruker fleire trær og tek gjennomsnittet av svara [2].

Det blei prøvd tre alpha-verdiar for ridge og tre djupner for kvar av tremodellane. Skogen har 48 trær, fire tilfeldige felt per splitt og minst fire rader per blad. Skogen med djupn 12 fekk lågast validerings-MAE, 0,510. Beste ridge fekk 0,521, så forskjellen er liten. Alle forsøka ligg i notebooken og results/metrics.json.

<!-- PAGEBREAK -->

## 3 Resultat

Den valde modellen blei testa utan ny trening på valideringsdataa.

| Modell | Test MAE | Test RMSE | Test R² | Innanfor 1 poeng |
|---|---|---|---|---|
| Median | 0,654 | 0,899 | -0,182 | 94,5 % |
| Random forest | 0,474 | 0,623 | 0,433 | 90,1 % |

Skogen fekk lågare gjennomsnittleg feil enn medianen. Medianen fekk likevel fleire svar innanfor eitt poeng, sidan mange vinar har vanlege vurderingar nær midten. Denne prosentdelen er derfor ikkje nok til å velje modell.

Trenings-MAE var 0,300, mot 0,510 på validering. Det tyder på noko overtilpassing. Dei største feila var på sjeldne kvalitetar. For kvalitet 3 og 8 var det berre tre testvinar kvar, så tala er usikre. Alkohol gav størst auke i valideringsfeil då feltet blei stokke om. Det viser at modellen bruker feltet, men beviser ikkje ein årsakssamanheng.

## 4 Deployment

Den ferdige modellen blir lagra i JSON og køyrd i nettlesaren. Målingane blir ikkje sende til ein prediksjonsserver. Skiljet mellom trening og bruk er omtalt i Google sitt kurs om ML-system [3].

Lokalt blir sida starta med python -m http.server 8000 --directory dist. Ho stoppar ugyldige verdiar og varslar om målingar utanfor treningsintervallet. Fem Python-testar og samanlikning av alle 272 testprediksjonane mellom Python og JavaScript bestod. Testane bestod også i GitHub Actions. Nettsida blei prøvd med endra verdiar og nullstilling.

Modellen bør testast på fleire vinar før praktisk bruk. Ved vidare drift bør ein kontrollere at modellen lastar og måle feil på nye vinar med kjende vurderingar. Slik overvaking er ikkje implementert i demoen.

## 5 Referansar

[1] Cortez mfl. (2009). [Wine Quality, UCI](https://doi.org/10.24432/C56S3T). CC BY 4.0.

[2] scikit-learn. [Decision trees](https://scikit-learn.org/stable/modules/tree.html) og [ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html). Fagkjelder; biblioteket er ikkje brukt i køyringa.

[3] Google. [Production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems). Fleire kjelder ligg i docs/SOURCES.md.