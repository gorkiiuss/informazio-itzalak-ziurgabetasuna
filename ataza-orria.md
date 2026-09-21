# GrALa lortzeko ataza-horria

GrALerako egin behar den lana ML eredu batek bere entrenamenduan datu multzoren bat ikasarazi ez bazaio, hauek berak inferitzerakoan zer nolako emaitzak lortzen diren aztertzean datza. Helburua hainbat datu mota eta inferentzia eredu aztergai izatea da haien harteko konbinaketa desberdinek zer emaitza ematen duten aztertzeko.

## 1. Esperimentua - Funtzio matematiko sinpleak

Momentuz, proiektuaren lehen fasea delako, aztertzeko erabiliko ditugun datuak sintetikoak izango dira. Lehenengo esperimentu bezala, funtzio matematiko batzuetatik laginak aterako dira, auzazkotasuna injektatuko zaie eta lagin osotik era desberdinetan zerikusia duten datuak, definizioz, informazioa dei diezaiokena, kenduko dira. Gero, inferentzia ereduek era desberdinetan ebakitako informazioarekin lortutako asmatze-tasak konparatuko dira.

### 1.1 Esperimentua - Sinpleena

Esperimenturik errazena funtzio lineala eta erregresio lineala erabiltzea da. Eta hori izango da lehen ataza/esperimentua. Helburu honetaz baliatuta proiektu osoa egiteko erabilgarria izango den sistema sortuko da. Programa pythonez kodetuko da. Lehenik eta behin datuak sortzeko sistema behar da. Honi "Data Generator" deituko zaio. Zein motatako data sortu behar duen, laginaren tamaina, izango duen auzazkotasuna eta ebakiko den informazioa. "Data Generator"-ak "Data" objetu desberdinak sortuko ditu, bere sortze-parametroak aldatuz, eta banan-banan sartuko dira inferentzia sisteman, non modelo desberdinen "Data" bakoitzerako asmatze-tasak kalkulatuko diren. "Data" guztien emaitzak gordeko dira. Sistema oso honi "Information-gap loss calculator" deituko zaio eta kalkuluak eragiteko behar dituen parametroak, Ebaluatuko diren datu motak, lagin tamaina desberdinak, auzazkotasun desberdinak eta ebakiko den informazio itzal desberdinak pasako zaiziko.

Informazio itzalak, ala "Information Shadows", lehen azaldu den bezala, haien artean zerikusia duten datu multzoak dira. Bi dimentsioko funtzio matematikoetan behintzat, azalera jarrai eta txiki baten gainean aurki daitezken puntu multzoa dira. Argi dago beraz, planoaren zati txiki eta itxi bat definitzen duen funtzio batek informazio kontzeptu bat definituko duela. Dena dela, lehenengo esperimentua ez zailtzeko asmoz, azalera multzo murriztu bat erabiliko da, etorkizun batean definitutako sistema dinamikoarekin bateragarria eginez. Momentuz, X zein Y ardatzen tarteekin, laukizuzenekin eta borobilekin egingo da lan.

"Information-gap loss calculator"-ak idatzitako emaitzak modulo batean kalkulatuko ditu eta beste batean haien bistaraketa sortuko da herrminta desberdinak erabiliz.

Hurrengo urratsak:

Zihurgabetasun neurria ondo ematen duen. Itzal bat tokatzen denean? Sare neuronal bat entrenatzen denean normalean irteeran 10 klase baditu.

Bayesian Neural Network with Dropout - 100 inferentzia egin, batazbestekoa inferentzia eta bariantza dugu ezegonkortasun marka. Agian bi geruzaro jarri, entreinatzeko erraz izateko? Konbergitu dezala, loss hori jaitzi overfittinga gainditzeko!

Bagging, esemble regressor classifiers. Intzerdidumbre neurria lortzeko (KNN, SVM..., Random Forestarekin).

Explorazioa bukatu eta experimentuak mugatu:

1. Random Forest - Azpitik egiten duen baggina lortu -> 1h 45min RandomForest + Azpiko infraestruktura
2. Poly 2 - Ensemble Bagging
3. NN + Dropout (Lehentasuna!!!)
4. Bayesian Neural Network.
    - BBB with TensorFlow
    - PBP

Irudikatzeko

Gardentasuna distribuzio normala?
2 grafikak: bataz bestekoa eta desbideraketa.

**26/02/20**:
 - Orain dropout handiagoarekin saiatu.
 - Bayesian Neural Networkekin hasi.
 - 

**2026/03/13**:
Azkenengo zutabea erabili gabe (klaserik gabe). Clusterrak egin, beste clusterretatik hurbilen dagoena kendu ziurgabetasuna kalkulatzeko. Aurreprozesaketa modura, dimentsio askotako itzala sortzeko prozedura gisa. K-Means, k hiperparametro bezala, optimizatu al silhouette index erabili? elbow method!

One-hot encoder ala balio nominalak baztertu?

Weka-ren 4pa bost datubaserekin frogak egin.

VSCode Latex - texlive-base, texlive-fonts-recommended, texlive-spanish, texlive-france, pdflatex