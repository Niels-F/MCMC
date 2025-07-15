import pexpect
import os
import sys
import numpy as np
from astropy.io import ascii
import time
import csv
from multiprocessing import Process
import re
import ast

start = time.time()

#######################
#        INPUT        #
#######################

file_path_config = '/home/nfaucher/Bureau/config.txt'




range_parameters = {}
parameters = []

range_pattern = re.compile(r'(\w+)_range\s*=\s*(.*)')
comment_pattern = re.compile(r'#.*')

with open(file_path_config, 'r') as ini_file_1:
    for line in ini_file_1:
        line = line.strip()
        if 'parameters' in line:  
            # Utiliser une nouvelle variable pour éviter de modifier 'line'
            line_parts = line.replace('[', '').replace(']', '').split('=')
            para = line_parts[1].strip()
            line_2 = para.split(',')
            for param in line_2:
                param = param.strip().strip("'")
                if param and not param.startswith('#'):
                    parameters.append(param)
                    
with open(file_path_config, 'r') as ini_file_2:
    for line in ini_file_2:
        line = line.strip()
        
        # Skip comments
        if comment_pattern.match(line):
            continue
         
        range_match = range_pattern.match(line)
        if range_match:
            param_name = range_match.group(1)
            if param_name in parameters:
                values = [float(v.strip()) for v in range_match.group(2).split(',') if v.strip()]
                range_parameters[param_name] = values
        if line.startswith('prob_ini'):
            line = line.split('=')
            prob_ini_str = line[1].strip()
            prob_ini_0 = ast.literal_eval(prob_ini_str)
        #if 'output' in line:
         #   line_val = line.split('=')
          #  val3_value = line_val[1].strip()

            

            
print("\nRange Parameters:")
print('saut---parameters',parameters)
print('saut--range-parameters',range_parameters)


def prob_ini(parameters,range_parameters):
    prob_ini = prob_ini_0 
    for param in parameters :
        prob_ini [param] = np.random.uniform(range_parameters[param][0], range_parameters[param][1])
    return prob_ini

    
y=prob_ini(parameters,range_parameters)
print('saut--prob_ini',y)
val3 = 'req,j2,j4'#val3_value
print('saut--val',val3)

############################
#   INITIALS CONSTRAINTS   #
############################

REQ_desired = 2.5559e9
sigma_req = 4e5
J2_desired = 3.51068e-3
sigma_j2 = 0.7e-6
J4_desired = -3.417e-5
sigma_j4 = 1.30e-6
############################
#   INITIALS PARAMETERS    #
############################


proposal_stddev_small = {'mnoyau': 0.001,'zatm_i': 0.001, 'zatm_r': 0.001, 'zdeep_i': 0.001, 'zdeep_r': 0.001,  'ppt':0.005}
proposal_stddev_big = {'mnoyau': 0.1,'zatm_i': 0.01, 'zatm_r': 0.01,'zdeep_i': 0.01, 'zdeep_r': 0.01,  'ppt':0.05}
proba_loi_instru_1 = 0.05
proba_loi_instru_2 = 0.6


#####################
#   RESULTS PATH    #
#####################
           
repo_dir = os.path.join(os.path.dirname(os.getcwd()), 'my_repository')
os.makedirs(repo_dir, exist_ok=True)

results_file_path = os.path.join(repo_dir, 'results.txt')
results_file_path_2 = os.path.join(repo_dir, 'results_2.txt')


# File handling functions
def read_ini_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def write_ini_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

############################
#       UPDATING FILE      #
############################

def update_ini_file(mnoyau_value, zatm_i_value, zatm_r_value, zdeep_i_value, zdeep_r_value,  ppt_value, file_path='jup.ini', file_path_2='cepam_etat.don'):
    lines = read_ini_file(file_path)
    lines2 = read_ini_file(file_path_2)

    # Update the main ini file
    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip().startswith('mnoyau ='):
                parts = line.strip().split('=')
                parts[1] = f' {mnoyau_value}'
                file.write('='.join(parts) + '\n')
            elif line.strip().startswith('zatm ='):
                parts = line.strip().split('=')
                values_and_comment = parts[1].split('{')
                # Récupérer les valeurs numériques
                values = values_and_comment[0].split(',')
                # Mettre à jour les valeurs numériques
                values[0] = f' {zatm_i_value}'
                values[1] = f' {zatm_r_value}'
                # Réassembler la ligne avec les nouvelles valeurs et le commentaire
                comment = '{' + '{'.join(values_and_comment[1:]).strip() if len(values_and_comment) > 1 else ''
                file.write(f'zatm = {",".join(values)} {comment}\n')
            elif line.strip().startswith('zdeep ='):
                parts = line.strip().split('=')
                values_and_comment = parts[1].split('{')
                # Récupérer les valeurs numériques
                values = values_and_comment[0].split(',')
                # Mettre à jour les valeurs numériques
                values[0] = f' {zdeep_i_value}'
                values[1] = f' {zdeep_r_value}'
                # Réassembler la ligne avec les nouvelles valeurs et le commentaire
                comment = '{' + '{'.join(values_and_comment[1:]).strip() if len(values_and_comment) > 1 else ''
                file.write(f'zdeep = {",".join(values)} {comment}\n')
            
            else:
                file.write(line)

    # Update the secondary file with ppt_value
    pattern = re.compile(r'^(p_ppt\s*=\s*)(\d*\.?\d*)\s*/?$')
    with open(file_path_2, 'w') as file2:
        for line in lines2:
            match = pattern.match(line.strip())
            if match:
                new_line = f"{match.group(1)}{ppt_value} /"
                file2.write(new_line + '\n')
            else:
                file2.write(line)

                
############################
#        RUN CEPAM         #
############################

def run_cepam():
    cepam_path = '../Testfolder_SVN/bin/cepam'
    try:
        child = pexpect.spawn(cepam_path, encoding='utf-8')
        child.logfile = sys.stdout
        child.expect(r'\(" "\-\> programme en mode interactif')
        child.sendline('')
        child.expect(r'\d \-\> Static calculation from an initial model from a binary file')
        child.sendline('2')
        child.expect(r'entrer le nom du modele a construire \(defaut: zeus\[\.don\]\)')
        child.sendline('jup')
        child.expect(r'Calcul du potentiel centrifuge perturbateur \(o/n\)\?')
        child.sendline('o')
        child.expect(r'faut\-il ecrire les modeles sur l\'ecran \(o/n\)\?')
        child.sendline('n')
        child.wait()
    except pexpect.ExceptionPexpect as e:
        print(f"Error running CEPAM: {e}")
     
def run_cepam_with_timeout(timeout=7):
    process = Process(target=run_cepam)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        print(f"Time de calcul trop long: {timeout} secondes. Relancer avec de nouveaux paramètres.")
        return False
    return True

############################
#      EXTRACT RESULT      #
############################

def extract_results():
    js_file_path = os.path.join(os.getcwd(), 'jup_js.csv')
    ini_file_path = os.path.join(os.getcwd(), 'jup.ini')
    ini_file_path_2 = os.path.join(os.getcwd(), 'cepam_etat.don')
    
    REQ_current, J2_current, J4_current = None, None, None
    mnoyau, zatm_i, zatm_r, zdeep_i, zdeep_r, ppt = None, None, None, None, None, None

    # Extraction from CSV file
    if os.path.exists(js_file_path):
        df = ascii.read(js_file_path, format='csv', fast_reader={'exponent_style': 'D'})
        try:
            REQ_current = float(df['REQ_CM'][0])
            print("req_curr", REQ_current)
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error reading REQ_proposal from {js_file_path}: {e}")
        try:
            J2_current = float(df['J2'][0])
            print("j2_curr", J2_current)
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error reading J2_proposal from {js_file_path}: {e}")
        try:
            J4_current = float(df['J4'][0])
            print("j4_curr", J4_current)
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error reading J4_proposal from {js_file_path}: {e}")

    # Extraction from jup.ini
    if os.path.exists(ini_file_path):
        with open(ini_file_path, 'r') as ini_file:
            for line in ini_file:
                if line.strip().startswith('mnoyau ='):
                    parts = line.strip().split('=')
                    try:
                        mnoyau = float(parts[1].strip().split(',')[0].strip())
                        print("extract", mnoyau)
                    except ValueError as e:
                        print(f"Error converting '{parts[1]}' to float: {e}")
                
                elif line.strip().startswith('zatm ='):
                    parts = line.strip().split('=')
                    try:
                        values_and_comment = parts[1].split('{', 1)
                        # Extraire uniquement les valeurs numérique
                        values = values_and_comment[0].strip().split(',')
                        zatm_i = float(values[0].strip())
                        zatm_r = float(values[1].strip())
                        print("Extracted zatm_i:", zatm_i)
                        print("Extracted zatm_r:", zatm_r)
                    except ValueError as e:
                        print(f"Error converting '{parts[1]}' to float: {e}")
                
                elif line.strip().startswith('zdeep ='):
                    parts = line.strip().split('=')
                    try:
                        values_and_comment = parts[1].split('{', 1)
                        # Extraire uniquement les valeurs numériques
                        values = values_and_comment[0].strip().split(',')
                        zdeep_i = float(values[0].strip())
                        zdeep_r = float(values[1].strip())
                        print("Extracted zdeep_i:", zdeep_i)
                        print("Extracted zdeep_r:", zdeep_r)
                    except ValueError as e:
                        print(f"Error converting '{parts[1]}' to float: {e}")

    # Extraction from cepam_etat.don
    if os.path.exists(ini_file_path_2):
        pattern = re.compile(r'^(p_ppt\s*=\s*)(\d*\.?\d*)\s*/?$')
        with open(ini_file_path_2, 'r') as ini_file_2:
            for line in ini_file_2:
                print("Reading line from cepam_etat.don:", line.strip())
                match = pattern.match(line.strip())
                if match:
                    ppt_str = match.group(2).strip()  # Extract only the second group
                    try:
                        ppt = float(ppt_str)
                        print("Extracted ppt:", ppt)
                    except ValueError as e:
                        print(f"Error converting '{ppt_str}' to float: {e}")

    return REQ_current, J2_current, J4_current, mnoyau, zatm_i, zatm_r, zdeep_i, zdeep_r, ppt


######################################
# CREATE A OUTPUT FILE W NEW VALUES  #
######################################

def append_results_to_file(output_file_path, index, REQ_CM, J2, J4, mnoyau_save, zatm_i_save, zatm_r_save, zdeep_i_save, zdeep_r_save, ppt_save, likelihood_proposal):
    with open(output_file_path, 'a') as output_file:
        output_file.write(f'Iteration {index}:\n')
        
        if REQ_CM is not None:
            output_file.write(f'REQ_CM = {REQ_CM}\n')
        else:
            output_file.write('Error: REQ_CM not found.\n')
            
        if J2 is not None:
            output_file.write(f'J2 = {J2}\n')
        else:
            output_file.write('Error: J2 not found.\n')
            
        if J4 is not None:
            output_file.write(f'J4 = {J4}\n')
        else:
            output_file.write('Error: J4 not found.\n')

        if mnoyau_save is not None:
            output_file.write(f'mnoyau = {mnoyau_save}\n')
        else:
            output_file.write('Error: mnoyau not found.\n')
            
        if zatm_i_save is not None:
            output_file.write(f'zatm_i = {zatm_i_save}\n')
        else:
            output_file.write('Error: zatm not found.\n')

        if zdeep_i_save is not None:
            output_file.write(f'zdeep_i = {zdeep_i_save}\n')
        else:
            output_file.write('Error: zdeep not found.\n')
            
        if zatm_r_save is not None:
            output_file.write(f'zatm_r = {zatm_r_save}\n')
        else:
            output_file.write('Error: zatm not found.\n')

        if zdeep_r_save is not None:
            output_file.write(f'zdeep_r = {zdeep_r_save}\n')
        else:
            output_file.write('Error: zdeep not found.\n')
            
        if ppt_save is not None:
            output_file.write(f'ppt = {ppt_save}\n')
        else:
            output_file.write('Error: ppt not found.\n')

        if likelihood_proposal is not None:
            output_file.write(f'likelihood = {likelihood_proposal}\n')
            output_file.write('\n')
        else:
            output_file.write('Error: likelihood not found.\n')

###########################
#      LIKELIHOOD         #
###########################

def likelihood(REQ_current, J2_current, J4_current, REQ_desired, sigma_req, J2_desired, sigma_j2, J4_desired, sigma_j4):
    J2_diff = (J2_current - J2_desired) / sigma_j2
    REQ_diff = (REQ_current - REQ_desired) / sigma_req
    J4_diff = (J4_current - J4_desired) / sigma_j4
    log_prob_x = 1
    
    #if val3 == 'req,j2':
    #	log_prob_x = np.log(1/(2*np.pi)**0.5*sigma_req)+np.log(1/(2*np.pi)**0.5*sigma_j2)-0.5*(J2_diff**2 + REQ_diff**2)
    	
    if val3 == 'req,j2,j4':
        #prob_x = (1 / ((2 * np.pi)**0.5 * sigma_req)) * (1 / ((2 * np.pi)**0.5 * sigma_j2)) * np.exp(-0.5 * ((J2_diff**2) + (REQ_diff)**2))
    	log_prob_x = np.log(1/(2*np.pi)**0.5*sigma_req)+np.log(1/(2*np.pi)**0.5*sigma_j2)+np.log(1/(2*np.pi)**0.5*sigma_j4)-0.5*(J2_diff**2 + J4_diff**2+ REQ_diff**2)
    print('ln likelihood', log_prob_x)
    return log_prob_x

##########################
#         MCMC           #
##########################

def mcmc(proposal_stddev, REQ_desired, sigma_req, J2_desired, sigma_j2, J4_desired, sigma_j4, max_iterations=5000):
    #prob_current = prob_ini(parameters,range_parameters)
    #posterior = [prob_current]

    # Calculate REQ for the initial position
    likelihood_current=-3e5
    compteur=0
    while likelihood_current<-2e5 and compteur<500:
       prob_current = prob_ini(parameters,range_parameters)
       posterior = [prob_current]
       update_ini_file(prob_current['mnoyau'], prob_current['zatm_i'],prob_current['zatm_r'] ,prob_current['zdeep_i'], prob_current['zdeep_r'], prob_current['ppt'])
       print("saut---update", prob_current['mnoyau'],prob_current['zatm_i'],prob_current['zatm_r'] ,prob_current['zdeep_i'], prob_current['zdeep_r'], prob_current['ppt'])
       if not run_cepam_with_timeout():
           continue
       REQ_current, J2_current, J4_current, mnoyau_current, zatm_i_current, zatm_r_current, zdeep_i_current,  zdeep_r_current, ppt_current = extract_results()
       likelihood_current = likelihood(REQ_current, J2_current,J4_current, REQ_desired, sigma_req, J2_desired, sigma_j2,J4_desired, sigma_j4)
       print("saut ---likelihood_current",likelihood_current,"saut---REQ",REQ_current,"saut---J2",J2_current,"saut---J4",J4_current)
       compteur = compteur + 1
       print("saut----- redraw ",compteur)
       append_results_to_file(results_file_path_2, 0, REQ_current, J2_current, J4_current, mnoyau_current, zatm_i_current, zatm_r_current, zdeep_i_current, zdeep_r_current, ppt_current, likelihood_current)
    append_results_to_file(results_file_path, 0, REQ_current, J2_current,J4_current, mnoyau_current,zatm_i_current, zatm_r_current, zdeep_i_current, zdeep_r_current, ppt_current, likelihood_current)

    for i in range(1, max_iterations):
        # Suggest new position
        prob_proposal = prob_current.copy()
        for param in parameters:
            loi_instru = np.random.uniform(0, 1)
            if loi_instru < proba_loi_instru_1:
                print("saut uniform")
                prob_proposal[param] = np.random.uniform(range_parameters[param][0], range_parameters[param][1])
            elif proba_loi_instru_1 <= loi_instru <= proba_loi_instru_2:
                print("petit saut")
                prob_proposal[param] = np.random.normal(prob_current[param], proposal_stddev_small[param])
            else:
                print("GRAND saut")
                prob_proposal[param] = np.random.normal(prob_current[param], proposal_stddev_big[param])  

        update_ini_file(prob_proposal['mnoyau'], prob_proposal['zatm_i'],prob_proposal['zatm_r'] ,prob_proposal['zdeep_i'], prob_proposal['zdeep_r'], prob_proposal['ppt'])
        
        # Condition to avoid error problem with the uniform law
        if not run_cepam_with_timeout():
           continue             
        REQ_proposal, J2_proposal,J4_proposal, mnoyau_proposal, zatm_i_proposal, zatm_r_proposal, zdeep_i_proposal, zdeep_r_proposal, ppt_proposal = extract_results()

        # Compute the likelihoods

        likelihood_proposal = likelihood(REQ_proposal, J2_proposal,J4_proposal, REQ_desired, sigma_req, J2_desired, sigma_j2,J4_desired, sigma_j4)

        print("likelihood_proposal",likelihood_proposal)
        

        # Compute acceptance probability
        p_accept = np.exp(likelihood_proposal - likelihood_current)
        u = np.random.uniform(0, 1)
        accept = u < p_accept

        if accept:
            prob_current = prob_proposal
            REQ_current = REQ_proposal
            J2_current = J2_proposal
            J4_current = J4_proposal
            mnoyau_current = mnoyau_proposal
            zatm_i_current = zatm_i_proposal
            zdeep_i_current = zdeep_i_proposal
            zatm_r_current = zatm_r_proposal
            zdeep_r_current = zdeep_r_proposal
            ppt_current = ppt_proposal
            likelihood_current = likelihood_proposal
            append_results_to_file(results_file_path, i, REQ_current, J2_current,J4_current, mnoyau_current, zatm_i_current, zatm_r_current, zdeep_i_current, zdeep_r_current, ppt_current, likelihood_current)
            print('Model accepted')
        else:
            append_results_to_file(results_file_path_2, 0, REQ_proposal, J2_proposal,J4_proposal, mnoyau_proposal, zatm_i_proposal, zatm_r_proposal, zdeep_i_proposal, zdeep_r_proposal, ppt_proposal, likelihood_proposal)
            print("Model rejected")
     

        print(f'Iteration {i} : ln Likelihood = {likelihood_current}, REQ = {REQ_current*1e-5:.6f}, J2 = {J2_current*1e6:.6f}, J4 = {J4_current*1e6:.6f},MNOYAU = {mnoyau_current:.6f}, Zatm_i = {zatm_i_current:.6f}, Zatm_r = {zatm_r_current:.6f}, Zdeep_i = {zdeep_i_current:.6f}, Zdeep_r = {zdeep_r_current:.6f} ,PPT = {ppt_current:.5f}, Probability accept = {p_accept}, U = {u:.5f}\n')
        
        print(f"Iteration {i}: Likelihood = {likelihood_proposal}, REQ = {REQ_proposal*1e-5:.6f}, J2 = {J2_proposal*1e6:.6f},J4 = {J4_proposal*1e6:.6f}, MNOYAU = {mnoyau_proposal:.6f}, Zatm_i = {zatm_i_proposal:.6f}, Zatm_r = {zatm_r_proposal:.6f}, Zdeep_i = {zdeep_i_proposal:.6f}, Zdeep_r = {zdeep_r_proposal:.6f} ,PPT = {ppt_proposal:.5f}\n")

        posterior.append(prob_current)

    return np.array(posterior)

##########################
#        RUN MCMC        #
##########################

posterior_samples = mcmc(proposal_stddev_small, REQ_desired, sigma_req, J2_desired, sigma_j2,J4_desired, sigma_j4)

end = time.time()
print("Time:", end - start)
# Votre script principal


