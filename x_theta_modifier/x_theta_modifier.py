# Add RDKit SA_Score to path for sascorer import
from x_theta_modifier.standard import StandardXThetaModifier
from x_theta_modifier.bon import BoNXThetaModifier
from x_theta_modifier.standard import StandardXThetaModifier
from x_theta_modifier.bon_local_search import BoNLocalSearchXThetaModifier
from x_theta_modifier.bon_local_search_last_step import BoNLocalSearchXThetaModifierLastStep

# Available sampling method classes
X_THETA_MODIFIER_CLASSES = {
    'standard': StandardXThetaModifier,
    'bon': BoNXThetaModifier,
    'bon_localsearch': BoNLocalSearchXThetaModifier,
    'bon_localsearch_laststep': BoNLocalSearchXThetaModifierLastStep,
}
