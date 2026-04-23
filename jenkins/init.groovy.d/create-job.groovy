import jenkins.model.*

def jenkins = Jenkins.get()
def jobs = new File("/var/jenkins_home/jobs")
if (!jobs.exists()) {
    jobs.mkdirs()
}

def jobName = "fraud-mlops-pipeline"
def jobDir = new File(jobs, jobName)
if (!jobDir.exists()) {
    jobDir.mkdirs()
}

def config = new File(jobDir, "config.xml")
config.text = """<flow-definition plugin="workflow-job">
  <actions/>
  <description>Local Jenkins pipeline for fraud MLOps assignment.</description>
  <keepDependencies>false</keepDependencies>
  <properties>
    <org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
      <triggers>
        <org.jenkinsci.plugins.gwt.GenericTrigger plugin="generic-webhook-trigger">
          <spec></spec>
          <token>fraud-alert-token</token>
          <printContributedVariables>true</printContributedVariables>
          <printPostContent>true</printPostContent>
          <silentResponse>false</silentResponse>
          <shouldNotFlattern>false</shouldNotFlattern>
          <allowSeveralTriggersPerBuild>false</allowSeveralTriggersPerBuild>
          <genericVariables>
            <org.jenkinsci.plugins.gwt.GenericVariable>
              <key>alert_name</key>
              <value>\$.alerts[0].labels.alertname</value>
              <expressionType>JSONPath</expressionType>
              <defaultValue>unknown</defaultValue>
              <regexpFilter></regexpFilter>
            </org.jenkinsci.plugins.gwt.GenericVariable>
            <org.jenkinsci.plugins.gwt.GenericVariable>
              <key>alert_state</key>
              <value>\$.alerts[0].status</value>
              <expressionType>JSONPath</expressionType>
              <defaultValue>unknown</defaultValue>
              <regexpFilter></regexpFilter>
            </org.jenkinsci.plugins.gwt.GenericVariable>
          </genericVariables>
          <genericRequestVariables/>
          <genericHeaderVariables/>
          <genericResponseVariables/>
          <regexpFilterText></regexpFilterText>
          <regexpFilterExpression></regexpFilterExpression>
        </org.jenkinsci.plugins.gwt.GenericTrigger>
      </triggers>
    </org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
  </properties>
  <definition class="org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition" plugin="workflow-cps">
    <script><![CDATA[
pipeline {
  agent any
  stages {
    stage('Context') {
      steps {
        echo 'Fraud MLOps Jenkins pipeline started'
        echo "Alert Name: ${env.alert_name}"
        echo "Alert State: ${env.alert_state}"
      }
    }
    stage('Validation') {
      steps {
        sh 'echo Running CI validation'
        sh 'python --version || true'
      }
    }
    stage('Retraining Trigger') {
      steps {
        sh 'echo Monitoring alert received, retraining workflow would be invoked here'
      }
    }
  }
}
    ]]></script>
    <sandbox>true</sandbox>
  </definition>
  <triggers/>
  <disabled>false</disabled>
</flow-definition>
"""

jenkins.reload()
