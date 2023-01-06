```shell
mkdir -p /opt/minecraft/instances/survival

cat >/lib/systemd/system/minecraft-survival.service<<'EOF'
[Unit]
Description=Minecraft Server
Wants=network.target
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/minecraft/instances/survival
ExecStart=/opt/java19/jdk-19.0.1/bin/java -Xms1G -Xmx1536m -jar /opt/minecraft/jars/server.jar nogui
RestartSec=30
Restart=on-failure
KillMode=process
KillSignal=SIGINT
SuccessExitStatus=130
StandardInput=null

[Install]
WantedBy=default.target
EOF


echo "eula=true" > /opt/minecraft/instances/survival/eula.txt

systemctl enable minecraft-survival.service

systemctl start minecraft-survival.service

systemctl status minecraft-survival.service

vim /opt/minecraft/instances/survival/server.properties

online-mode=false
server-port=9977
```

