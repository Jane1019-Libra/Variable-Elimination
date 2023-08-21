import numpy as np

'''
Implement the variable elimination algorithm by coding the
following functions in Python. Factors are essentially 
multi-dimensional arrays. Hence use numpy multidimensional 
arrays as your main data structure.  If you are not familiar 
with numpy, go through the following tutorial: 
https://numpy.org/doc/stable/user/quickstart.html
'''



######### restrict function
# Tip: Use slicing operations to implement this function
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be restricted
# value -- integer indicating the value to be assigned to variable
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been restricted to value)
#########
def restrict(factor,variable,value):
	new_form = factor.shape
	resulting_factor = factor.take(indices=value, axis=variable)
	new_form_list = list(new_form)
	new_form_list[variable] = 1
	resulting_factor = resulting_factor.reshape(tuple(new_form_list))
	return resulting_factor

######### sumout function
# Tip: Use numpy.sum to implement this function
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be summed out
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been summed out)
#########
def sumout(factor,variable):
	# dummy result until the function is filled in
	new_form = factor.shape
	resulting_factor = np.sum(factor, axis=variable)
	new_form_list = list(new_form)
	new_form_list[variable] = 1
	resulting_factor = resulting_factor.reshape(tuple(new_form_list))
	return resulting_factor

######### multiply function
# Tip: take advantage of numpy broadcasting rules to multiply factors with different variables
# See https://numpy.org/doc/stable/user/basics.broadcasting.html
#
# Inputs: 
# factor1 -- multidimensional array (one dimension per variable in the domain)
# factor2 -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (elementwise product of the two factors)
#########
def multiply(factor1,factor2):

	# dummy result until the function is filled in
	resulting_factor = np.multiply(factor1, factor2)
	return resulting_factor

######### normalize function
# Tip: divide by the sum of all entries to normalize the factor
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (entries are normalized to sum up to 1)
#########
def normalize(factor):
	# dummy result until the function is filled in
	sum = np.sum(factor)
	resulting_factor = np.divide(factor, sum)
	return resulting_factor

######### inference function
# Tip: function that computes Pr(query_variables|evidence_list) by variable elimination.  
# This function should restrict the factors in factor_list according to the
# evidence in evidence_list.  Next, it should sumout the hidden variables from the 
# product of the factors in factor_list.  The variables should be summed out in the 
# order given in ordered_list_of_hidden_variables.  Finally, the answer should be
# normalized to obtain a probability distribution that sums up to 1.
#
#Inputs: 
#factor_list -- list of factors (multidimensional arrays) that define the joint distribution of the domain
#query_variables -- list of variables (integers) for which we need to compute the conditional distribution
#ordered_list_of_hidden_variables -- list of variables (integers) that need to be eliminated according to thir order in the list
#evidence_list -- list of assignments where each assignment consists of a variable and a value assigned to it (e.g., [[var1,val1],[var2,val2]])
#
#Output:
#answer -- multidimensional array (conditional distribution P(query_variables|evidence_list))
#########
def inference(factor_list,query_variables,ordered_list_of_hidden_variables,evidence_list):
	l = len(factor_list)
	for each in evidence_list:
		variable = each[0]
		value = each[1]
		for i in range(l):
			shape_here = factor_list[i].shape
			if shape_here[variable] > 1:
				factor_list[i] = restrict(factor_list[i], variable, value)
				print(variable)
				print(shape_here)
				print(factor_list[i].shape)
				print(f"{np.squeeze(factor_list[i])}\n")

	for each in ordered_list_of_hidden_variables:
		i = 0
		need_list = []
		while i < len(factor_list):
			shape_here = factor_list[i].shape
			if shape_here[each] > 1:
				need_list.append(factor_list[i])
				del factor_list[i]
				i = i - 1
			i = i + 1
		if len(need_list) != 0:
			new_factor = need_list[0]
			for i in range(len(need_list) - 1):
				new_factor = multiply(new_factor, need_list[i+1])
			new_factor = sumout(new_factor, each)
			print(each)
			print(len(need_list))
			print(f"{np.squeeze(new_factor)}\n")
			factor_list.append(new_factor)
	
	for each in query_variables:
		i = 0
		need_list = []
		while i < len(factor_list):
			shape_here = factor_list[i].shape
			if shape_here[each] > 1:
					need_list.append(factor_list[i])
					del factor_list[i]
					i = i - 1
			i = i + 1
		
		if len(need_list) != 0:
			new_factor = need_list[0]
			for i in range(len(need_list) - 1):
				new_factor = multiply(new_factor, need_list[i+1])
			factor_list.append(new_factor)
			print(each)
			print(len(need_list))
			print(f"{np.squeeze(new_factor)}\n")

	final_len = len(factor_list)
	for i in range(final_len):
		factor_list[i] = normalize(factor_list[i])
	return factor_list

# Example Bayes net from the lecture slides: A -> B -> C

# variables
Trav=0
Fraud=1
FP=2
Acc=3
PT=4
OP=5

# factors

f1 = np.array([0.95,0.05])
f1 = f1.reshape(2,1,1,1,1,1)
print(f"f1={np.squeeze(f1)}\n")

f2 = np.array([[0.996,0.004],[0.99,0.01]])
f2 = f2.reshape(2,2,1,1,1,1)
print(f"f2={np.squeeze(f2)}\n")

f3 = np.array([[[0.99,0.01],[0.9,0.1]],[[0.1,0.9],[0.1,0.9]]])
f3 = f3.reshape(2,2,2,1,1,1)
print(f"f3={np.squeeze(f3)}\n")

f4 = np.array([0.2,0.8])
f4 = f4.reshape(1,1,1,2,1,1)
print(f"f4={np.squeeze(f4)}\n")

f5 = np.array([[0.99,0.01],[0.9,0.1]])
f5 = f5.reshape(1,1,1,2,2,1)
print(f"f5={np.squeeze(f5)}\n")

f6 = np.array([[[0.9,0.1],[0.4,0.6]],[[0.7,0.3],[0.2,0.8]]])
f6 = f6.reshape(1,2,1,2,1,2)
print(f"f6={np.squeeze(f6)}\n")

f7 = inference([f1,f2],[Fraud],[Trav],[])
print(f"P(Fraud)={np.squeeze(f7)}\n")

f8 = inference([f1,f2,f3,f4,f5,f6],[Fraud],[Trav,Acc],[[FP,1],[OP,0],[PT,1]])
print(f"P(Fraud|FP,~OP,PT)={np.squeeze(f8)}\n")

f9 = inference([f2,f3,f4,f5,f6],[Fraud],[Acc],[[Trav, 1],[PT,1],[OP,0],[FP, 1]])
print(f"P(Fraud|Trav,FP,~OP,PT)={np.squeeze(f9)}\n")

f10 = inference([f1,f2,f3,f4,f5,f6],[Fraud],[Trav,FP,Acc],[[PT, 1],[OP,1]])
print(f"P(Fraud|OP,PT)={np.squeeze(f10)}\n")

f11 = inference([f1,f2,f3,f4,f5,f6],[Fraud],[Trav,FP,Acc,PT],[[OP,1]])
print(f"P(Fraud|OP)={np.squeeze(f11)}\n")

'''
A=0
B=1
C=2
variables = np.array(['A','B','C'])

# values
false=0
true=1
values = np.array(['false','true'])

# factors

# Pr(A)
f1 = np.array([0.1,0.9])
f1 = f1.reshape(2,1,1)
print(f"Pr(A)={np.squeeze(f1)}\n")

# Pr(B|A)
f2 = np.array([[0.6,0.4],[0.1,0.9]])
f2 = f2.reshape(2,2,1)
print(f"Pr(B|A)={np.squeeze(f2)}\n")

# Pr(C|B)
f3 = np.array([[0.8,0.2],[0.3,0.7]])
f3 = f3.reshape(1,2,2)
print(f"Pr(C|B)={np.squeeze(f3)}\n")

f4 = np.array([[[0.6,0.4],[0.1,0.9]],[[0.8,0.2],[0.3,0.7]]])
f4 = f4.reshape(2,2,2)
print(f"Pr(A,B,C)={np.squeeze(f4)}\n")
f8 = restrict(f4,A,true)

# multiply two factors
f4 = multiply(f2,f3)
print(f"multiply(f2,f3)={np.squeeze(f4)}\n")

# sumout a variable
f5 = sumout(f2,A)
print(f"sumout(f2,A)={np.squeeze(f5)}\n")

# restricting a factor
f6 = restrict(f2,A,true)
print(f"restrict(f2,A,true)={np.squeeze(f6)}\n")

# inference P(C)
f7 = inference([f1,f2,f3],[C],[A,B],[])
print(f"P(C)={np.squeeze(f7)}\n")
'''
