# Sistemi-Operativi-Dedicati

È stata implementata sulla scheda STM32F429I-DISC1 di STMicroelectronics un’intelligenza artificiale costituita da una rete neurale basata su VGG, e utilizzata per la classificazione di immagini fornite alla scheda attraverso una pennetta USB. Il programma è composto da due modalità:
- una modalità di test che esegue la classificazione e calcola l'accuratezza del modello e il tempo di classificazione per ogni immagine
- una modalità demo che mostra sullo schermo un'immagine alla volta e mostra la predizione ottenuta
È possibile passare da una modalità all'altra in fase di avvio tramite la pressione del bottone blu (USER) integrato sulla scheda.
Nel repository sono presenti anche le immagini di test e demo, oltre che ai file utilizzati per il training del modello, effettuato utilizzando Google Colab e il dataset CIFAR10.
