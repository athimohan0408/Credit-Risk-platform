# Build stage
FROM node:18-alpine as build
WORKDIR /app

# Accept API URL as build arg (defaults to relative /api for Docker, overridden for dev)
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Serve stage with nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
