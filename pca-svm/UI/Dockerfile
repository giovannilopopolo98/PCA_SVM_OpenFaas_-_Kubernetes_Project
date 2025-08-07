FROM nginx:alpine

# Elimina tutto il contenuto default
RUN rm -rf /usr/share/nginx/html/*

# Copia solo i file specifici per il frontend
COPY index.html /usr/share/nginx/html/index.html
COPY style.css /usr/share/nginx/html/style.css
COPY script.js /usr/share/nginx/html/script.js
