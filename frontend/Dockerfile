# Use an official Node.js image as a parent image
FROM node:16 AS build

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to the container
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy only the specific directories and files needed for building
COPY . .

# Build the application
RUN npm run build

# Use a lightweight Nginx image for serving the static files
FROM nginx:1.21.6-alpine

# Copy the build output from the builder stage to the container
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx.conf file
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy env.sh that we can changing our envrionment variable for API
COPY env.sh /docker-entrypoint.d/env.sh
RUN chmod +x /docker-entrypoint.d/env.sh

# Expose port 80
EXPOSE 80

# Define the default command to run Nginx and serve the application
CMD ["nginx", "-g", "daemon off;"]
