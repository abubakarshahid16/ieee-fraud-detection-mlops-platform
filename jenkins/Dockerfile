FROM jenkins/jenkins:lts-jdk17

ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false"

USER root
RUN jenkins-plugin-cli --plugins \
    workflow-aggregator \
    job-dsl \
    configuration-as-code \
    generic-webhook-trigger

USER jenkins
