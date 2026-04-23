import hudson.security.FullControlOnceLoggedInAuthorizationStrategy
import jenkins.model.*
import jenkins.install.InstallState
import hudson.security.HudsonPrivateSecurityRealm

def instance = Jenkins.get()

def hudsonRealm = new HudsonPrivateSecurityRealm(false)
if (hudsonRealm.getUser("admin") == null) {
    hudsonRealm.createAccount("admin", "admin")
}
instance.setSecurityRealm(hudsonRealm)

def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
strategy.setAllowAnonymousRead(false)
instance.setAuthorizationStrategy(strategy)
instance.setInstallState(InstallState.INITIAL_SETUP_COMPLETED)
instance.save()
