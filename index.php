Hello world <br><br>

Czas serwera: <?= date("Y-m-d h:i:s", time()); ?><br>
Czas ostatniej zmiany pliku: <?=date("Y-m-d h:i:s",  filemtime(__FILE__));; ?>
