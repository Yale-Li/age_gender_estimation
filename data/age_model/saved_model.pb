
ò
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

û
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8ëÿ

Adam/gender_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/gender_output/bias/v

-Adam/gender_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/gender_output/bias/v*
_output_shapes
:*
dtype0

Adam/gender_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/gender_output/kernel/v

/Adam/gender_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gender_output/kernel/v*
_output_shapes
:	*
dtype0

#Adam/batch_normalization_314/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_314/beta/v

7Adam/batch_normalization_314/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_314/beta/v*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_314/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_314/gamma/v

8Adam/batch_normalization_314/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_314/gamma/v*
_output_shapes	
:*
dtype0

Adam/dense_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_69/bias/v
z
(Adam/dense_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_69/kernel/v

*Adam/dense_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/v* 
_output_shapes
:
*
dtype0

#Adam/batch_normalization_313/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_313/beta/v

7Adam/batch_normalization_313/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_313/beta/v*
_output_shapes
:@*
dtype0
 
$Adam/batch_normalization_313/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_313/gamma/v

8Adam/batch_normalization_313/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_313/gamma/v*
_output_shapes
:@*
dtype0

Adam/conv2d_442/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_442/bias/v
}
*Adam/conv2d_442/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_442/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_442/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_442/kernel/v

,Adam/conv2d_442/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_442/kernel/v*&
_output_shapes
: @*
dtype0

#Adam/batch_normalization_312/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_312/beta/v

7Adam/batch_normalization_312/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_312/beta/v*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_312/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_312/gamma/v

8Adam/batch_normalization_312/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_312/gamma/v*
_output_shapes
: *
dtype0

Adam/conv2d_441/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_441/bias/v
}
*Adam/conv2d_441/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_441/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_441/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_441/kernel/v

,Adam/conv2d_441/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_441/kernel/v*&
_output_shapes
: *
dtype0

#Adam/batch_normalization_311/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_311/beta/v

7Adam/batch_normalization_311/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_311/beta/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_311/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_311/gamma/v

8Adam/batch_normalization_311/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_311/gamma/v*
_output_shapes
:*
dtype0

Adam/conv2d_440/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_440/bias/v
}
*Adam/conv2d_440/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_440/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_440/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_440/kernel/v

,Adam/conv2d_440/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_440/kernel/v*&
_output_shapes
:*
dtype0

Adam/gender_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/gender_output/bias/m

-Adam/gender_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/gender_output/bias/m*
_output_shapes
:*
dtype0

Adam/gender_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/gender_output/kernel/m

/Adam/gender_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gender_output/kernel/m*
_output_shapes
:	*
dtype0

#Adam/batch_normalization_314/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_314/beta/m

7Adam/batch_normalization_314/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_314/beta/m*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_314/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_314/gamma/m

8Adam/batch_normalization_314/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_314/gamma/m*
_output_shapes	
:*
dtype0

Adam/dense_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_69/bias/m
z
(Adam/dense_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_69/kernel/m

*Adam/dense_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/m* 
_output_shapes
:
*
dtype0

#Adam/batch_normalization_313/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_313/beta/m

7Adam/batch_normalization_313/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_313/beta/m*
_output_shapes
:@*
dtype0
 
$Adam/batch_normalization_313/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_313/gamma/m

8Adam/batch_normalization_313/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_313/gamma/m*
_output_shapes
:@*
dtype0

Adam/conv2d_442/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_442/bias/m
}
*Adam/conv2d_442/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_442/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_442/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_442/kernel/m

,Adam/conv2d_442/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_442/kernel/m*&
_output_shapes
: @*
dtype0

#Adam/batch_normalization_312/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_312/beta/m

7Adam/batch_normalization_312/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_312/beta/m*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_312/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_312/gamma/m

8Adam/batch_normalization_312/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_312/gamma/m*
_output_shapes
: *
dtype0

Adam/conv2d_441/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_441/bias/m
}
*Adam/conv2d_441/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_441/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_441/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_441/kernel/m

,Adam/conv2d_441/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_441/kernel/m*&
_output_shapes
: *
dtype0

#Adam/batch_normalization_311/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_311/beta/m

7Adam/batch_normalization_311/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_311/beta/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_311/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_311/gamma/m

8Adam/batch_normalization_311/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_311/gamma/m*
_output_shapes
:*
dtype0

Adam/conv2d_440/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_440/bias/m
}
*Adam/conv2d_440/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_440/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_440/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_440/kernel/m

,Adam/conv2d_440/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_440/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
|
gender_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namegender_output/bias
u
&gender_output/bias/Read/ReadVariableOpReadVariableOpgender_output/bias*
_output_shapes
:*
dtype0

gender_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_namegender_output/kernel
~
(gender_output/kernel/Read/ReadVariableOpReadVariableOpgender_output/kernel*
_output_shapes
:	*
dtype0
§
'batch_normalization_314/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_314/moving_variance
 
;batch_normalization_314/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_314/moving_variance*
_output_shapes	
:*
dtype0

#batch_normalization_314/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_314/moving_mean

7batch_normalization_314/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_314/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_314/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_314/beta

0batch_normalization_314/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_314/beta*
_output_shapes	
:*
dtype0

batch_normalization_314/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_314/gamma

1batch_normalization_314/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_314/gamma*
_output_shapes	
:*
dtype0
s
dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_69/bias
l
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes	
:*
dtype0
|
dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_69/kernel
u
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel* 
_output_shapes
:
*
dtype0
¦
'batch_normalization_313/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_313/moving_variance

;batch_normalization_313/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_313/moving_variance*
_output_shapes
:@*
dtype0

#batch_normalization_313/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_313/moving_mean

7batch_normalization_313/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_313/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_313/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_313/beta

0batch_normalization_313/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_313/beta*
_output_shapes
:@*
dtype0

batch_normalization_313/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_313/gamma

1batch_normalization_313/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_313/gamma*
_output_shapes
:@*
dtype0
v
conv2d_442/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_442/bias
o
#conv2d_442/bias/Read/ReadVariableOpReadVariableOpconv2d_442/bias*
_output_shapes
:@*
dtype0

conv2d_442/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_442/kernel

%conv2d_442/kernel/Read/ReadVariableOpReadVariableOpconv2d_442/kernel*&
_output_shapes
: @*
dtype0
¦
'batch_normalization_312/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_312/moving_variance

;batch_normalization_312/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_312/moving_variance*
_output_shapes
: *
dtype0

#batch_normalization_312/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_312/moving_mean

7batch_normalization_312/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_312/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_312/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_312/beta

0batch_normalization_312/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_312/beta*
_output_shapes
: *
dtype0

batch_normalization_312/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_312/gamma

1batch_normalization_312/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_312/gamma*
_output_shapes
: *
dtype0
v
conv2d_441/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_441/bias
o
#conv2d_441/bias/Read/ReadVariableOpReadVariableOpconv2d_441/bias*
_output_shapes
: *
dtype0

conv2d_441/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_441/kernel

%conv2d_441/kernel/Read/ReadVariableOpReadVariableOpconv2d_441/kernel*&
_output_shapes
: *
dtype0
¦
'batch_normalization_311/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_311/moving_variance

;batch_normalization_311/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_311/moving_variance*
_output_shapes
:*
dtype0

#batch_normalization_311/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_311/moving_mean

7batch_normalization_311/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_311/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_311/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_311/beta

0batch_normalization_311/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_311/beta*
_output_shapes
:*
dtype0

batch_normalization_311/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_311/gamma

1batch_normalization_311/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_311/gamma*
_output_shapes
:*
dtype0
v
conv2d_440/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_440/bias
o
#conv2d_440/bias/Read/ReadVariableOpReadVariableOpconv2d_440/bias*
_output_shapes
:*
dtype0

conv2d_440/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_440/kernel

%conv2d_440/kernel/Read/ReadVariableOpReadVariableOpconv2d_440/kernel*&
_output_shapes
:*
dtype0

serving_default_input_56Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  
Í
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_56conv2d_440/kernelconv2d_440/biasbatch_normalization_311/gammabatch_normalization_311/beta#batch_normalization_311/moving_mean'batch_normalization_311/moving_varianceconv2d_441/kernelconv2d_441/biasbatch_normalization_312/gammabatch_normalization_312/beta#batch_normalization_312/moving_mean'batch_normalization_312/moving_varianceconv2d_442/kernelconv2d_442/biasbatch_normalization_313/gammabatch_normalization_313/beta#batch_normalization_313/moving_mean'batch_normalization_313/moving_variancedense_69/kerneldense_69/bias'batch_normalization_314/moving_variancebatch_normalization_314/gamma#batch_normalization_314/moving_meanbatch_normalization_314/betagender_output/kernelgender_output/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1229797

NoOpNoOp
ó
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*­
value¢B B
þ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
È
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op*
Õ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance*

-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
È
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op*
Õ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance*

G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
È
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op*
Õ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\axis
	]gamma
^beta
_moving_mean
`moving_variance*

a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 

g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
¦
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias*
Õ
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{axis
	|gamma
}beta
~moving_mean
moving_variance*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
®
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ì
0
 1
)2
*3
+4
,5
96
:7
C8
D9
E10
F11
S12
T13
]14
^15
_16
`17
s18
t19
|20
}21
~22
23
24
25*

0
 1
)2
*3
94
:5
C6
D7
S8
T9
]10
^11
s12
t13
|14
}15
16
17*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
µ
	iter
beta_1
beta_2

decay
 learning_ratem m)m*m9m:mCmDm Sm¡Tm¢]m£^m¤sm¥tm¦|m§}m¨	m©	mªv« v¬)v­*v®9v¯:v°Cv±Dv²Sv³Tv´]vµ^v¶sv·tv¸|v¹}vº	v»	v¼*

¡serving_default* 

0
 1*

0
 1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

§trace_0* 

¨trace_0* 
a[
VARIABLE_VALUEconv2d_440/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_440/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
)0
*1
+2
,3*

)0
*1*
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

®trace_0
¯trace_1* 

°trace_0
±trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_311/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_311/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_311/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_311/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

·trace_0* 

¸trace_0* 

90
:1*

90
:1*
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

¾trace_0* 

¿trace_0* 
a[
VARIABLE_VALUEconv2d_441/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_441/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
C0
D1
E2
F3*

C0
D1*
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

Åtrace_0
Ætrace_1* 

Çtrace_0
Ètrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_312/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_312/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_312/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_312/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

Îtrace_0* 

Ïtrace_0* 

S0
T1*

S0
T1*
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

Õtrace_0* 

Ötrace_0* 
a[
VARIABLE_VALUEconv2d_442/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_442/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
]0
^1
_2
`3*

]0
^1*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

Ütrace_0
Ýtrace_1* 

Þtrace_0
ßtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_313/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_313/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_313/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_313/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

åtrace_0* 

ætrace_0* 
* 
* 
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

ìtrace_0* 

ítrace_0* 

s0
t1*

s0
t1*
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

ótrace_0* 

ôtrace_0* 
_Y
VARIABLE_VALUEdense_69/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_69/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
|0
}1
~2
3*

|0
}1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

útrace_0
ûtrace_1* 

ütrace_0
ýtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_314/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_314/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_314/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_314/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
d^
VARIABLE_VALUEgender_output/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEgender_output/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
<
+0
,1
E2
F3
_4
`5
~6
7*
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

E0
F1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

_0
`1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

~0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
~
VARIABLE_VALUEAdam/conv2d_440/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_440/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_311/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_311/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_441/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_441/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_312/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_312/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_442/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_442/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_313/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_313/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_69/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_69/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_314/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_314/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/gender_output/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/gender_output/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_440/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_440/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_311/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_311/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_441/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_441/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_312/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_312/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_442/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_442/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_313/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_313/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_69/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_69/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_314/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_314/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/gender_output/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/gender_output/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_440/kernel/Read/ReadVariableOp#conv2d_440/bias/Read/ReadVariableOp1batch_normalization_311/gamma/Read/ReadVariableOp0batch_normalization_311/beta/Read/ReadVariableOp7batch_normalization_311/moving_mean/Read/ReadVariableOp;batch_normalization_311/moving_variance/Read/ReadVariableOp%conv2d_441/kernel/Read/ReadVariableOp#conv2d_441/bias/Read/ReadVariableOp1batch_normalization_312/gamma/Read/ReadVariableOp0batch_normalization_312/beta/Read/ReadVariableOp7batch_normalization_312/moving_mean/Read/ReadVariableOp;batch_normalization_312/moving_variance/Read/ReadVariableOp%conv2d_442/kernel/Read/ReadVariableOp#conv2d_442/bias/Read/ReadVariableOp1batch_normalization_313/gamma/Read/ReadVariableOp0batch_normalization_313/beta/Read/ReadVariableOp7batch_normalization_313/moving_mean/Read/ReadVariableOp;batch_normalization_313/moving_variance/Read/ReadVariableOp#dense_69/kernel/Read/ReadVariableOp!dense_69/bias/Read/ReadVariableOp1batch_normalization_314/gamma/Read/ReadVariableOp0batch_normalization_314/beta/Read/ReadVariableOp7batch_normalization_314/moving_mean/Read/ReadVariableOp;batch_normalization_314/moving_variance/Read/ReadVariableOp(gender_output/kernel/Read/ReadVariableOp&gender_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_440/kernel/m/Read/ReadVariableOp*Adam/conv2d_440/bias/m/Read/ReadVariableOp8Adam/batch_normalization_311/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_311/beta/m/Read/ReadVariableOp,Adam/conv2d_441/kernel/m/Read/ReadVariableOp*Adam/conv2d_441/bias/m/Read/ReadVariableOp8Adam/batch_normalization_312/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_312/beta/m/Read/ReadVariableOp,Adam/conv2d_442/kernel/m/Read/ReadVariableOp*Adam/conv2d_442/bias/m/Read/ReadVariableOp8Adam/batch_normalization_313/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_313/beta/m/Read/ReadVariableOp*Adam/dense_69/kernel/m/Read/ReadVariableOp(Adam/dense_69/bias/m/Read/ReadVariableOp8Adam/batch_normalization_314/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_314/beta/m/Read/ReadVariableOp/Adam/gender_output/kernel/m/Read/ReadVariableOp-Adam/gender_output/bias/m/Read/ReadVariableOp,Adam/conv2d_440/kernel/v/Read/ReadVariableOp*Adam/conv2d_440/bias/v/Read/ReadVariableOp8Adam/batch_normalization_311/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_311/beta/v/Read/ReadVariableOp,Adam/conv2d_441/kernel/v/Read/ReadVariableOp*Adam/conv2d_441/bias/v/Read/ReadVariableOp8Adam/batch_normalization_312/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_312/beta/v/Read/ReadVariableOp,Adam/conv2d_442/kernel/v/Read/ReadVariableOp*Adam/conv2d_442/bias/v/Read/ReadVariableOp8Adam/batch_normalization_313/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_313/beta/v/Read/ReadVariableOp*Adam/dense_69/kernel/v/Read/ReadVariableOp(Adam/dense_69/bias/v/Read/ReadVariableOp8Adam/batch_normalization_314/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_314/beta/v/Read/ReadVariableOp/Adam/gender_output/kernel/v/Read/ReadVariableOp-Adam/gender_output/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1230808
ø
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_440/kernelconv2d_440/biasbatch_normalization_311/gammabatch_normalization_311/beta#batch_normalization_311/moving_mean'batch_normalization_311/moving_varianceconv2d_441/kernelconv2d_441/biasbatch_normalization_312/gammabatch_normalization_312/beta#batch_normalization_312/moving_mean'batch_normalization_312/moving_varianceconv2d_442/kernelconv2d_442/biasbatch_normalization_313/gammabatch_normalization_313/beta#batch_normalization_313/moving_mean'batch_normalization_313/moving_variancedense_69/kerneldense_69/biasbatch_normalization_314/gammabatch_normalization_314/beta#batch_normalization_314/moving_mean'batch_normalization_314/moving_variancegender_output/kernelgender_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_440/kernel/mAdam/conv2d_440/bias/m$Adam/batch_normalization_311/gamma/m#Adam/batch_normalization_311/beta/mAdam/conv2d_441/kernel/mAdam/conv2d_441/bias/m$Adam/batch_normalization_312/gamma/m#Adam/batch_normalization_312/beta/mAdam/conv2d_442/kernel/mAdam/conv2d_442/bias/m$Adam/batch_normalization_313/gamma/m#Adam/batch_normalization_313/beta/mAdam/dense_69/kernel/mAdam/dense_69/bias/m$Adam/batch_normalization_314/gamma/m#Adam/batch_normalization_314/beta/mAdam/gender_output/kernel/mAdam/gender_output/bias/mAdam/conv2d_440/kernel/vAdam/conv2d_440/bias/v$Adam/batch_normalization_311/gamma/v#Adam/batch_normalization_311/beta/vAdam/conv2d_441/kernel/vAdam/conv2d_441/bias/v$Adam/batch_normalization_312/gamma/v#Adam/batch_normalization_312/beta/vAdam/conv2d_442/kernel/vAdam/conv2d_442/bias/v$Adam/batch_normalization_313/gamma/v#Adam/batch_normalization_313/beta/vAdam/dense_69/kernel/vAdam/dense_69/bias/v$Adam/batch_normalization_314/gamma/v#Adam/batch_normalization_314/beta/vAdam/gender_output/kernel/vAdam/gender_output/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1231031º


G__inference_conv2d_441_layer_call_and_return_conditional_losses_1229115

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228858

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


G__inference_conv2d_440_layer_call_and_return_conditional_losses_1229088

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_312_layer_call_fn_1230276

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228889
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Ã
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230220

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

/__inference_gender_output_layer_call_fn_1230561

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gender_output_layer_call_and_return_conditional_losses_1229210o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ã
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230404

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü
e
,__inference_dropout_72_layer_call_fn_1230535

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229302p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230294

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý	
f
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229302

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ã
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228813

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1230414

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
®
*__inference_model_53_layer_call_fn_1229911

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_53_layer_call_and_return_conditional_losses_1229480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ì

E__inference_model_53_layer_call_and_return_conditional_losses_1230014

inputsC
)conv2d_440_conv2d_readvariableop_resource:8
*conv2d_440_biasadd_readvariableop_resource:=
/batch_normalization_311_readvariableop_resource:?
1batch_normalization_311_readvariableop_1_resource:N
@batch_normalization_311_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_311_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_441_conv2d_readvariableop_resource: 8
*conv2d_441_biasadd_readvariableop_resource: =
/batch_normalization_312_readvariableop_resource: ?
1batch_normalization_312_readvariableop_1_resource: N
@batch_normalization_312_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_312_fusedbatchnormv3_readvariableop_1_resource: C
)conv2d_442_conv2d_readvariableop_resource: @8
*conv2d_442_biasadd_readvariableop_resource:@=
/batch_normalization_313_readvariableop_resource:@?
1batch_normalization_313_readvariableop_1_resource:@N
@batch_normalization_313_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_313_fusedbatchnormv3_readvariableop_1_resource:@;
'dense_69_matmul_readvariableop_resource:
7
(dense_69_biasadd_readvariableop_resource:	H
9batch_normalization_314_batchnorm_readvariableop_resource:	L
=batch_normalization_314_batchnorm_mul_readvariableop_resource:	J
;batch_normalization_314_batchnorm_readvariableop_1_resource:	J
;batch_normalization_314_batchnorm_readvariableop_2_resource:	?
,gender_output_matmul_readvariableop_resource:	;
-gender_output_biasadd_readvariableop_resource:
identity¢7batch_normalization_311/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_311/ReadVariableOp¢(batch_normalization_311/ReadVariableOp_1¢7batch_normalization_312/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_312/ReadVariableOp¢(batch_normalization_312/ReadVariableOp_1¢7batch_normalization_313/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_313/ReadVariableOp¢(batch_normalization_313/ReadVariableOp_1¢0batch_normalization_314/batchnorm/ReadVariableOp¢2batch_normalization_314/batchnorm/ReadVariableOp_1¢2batch_normalization_314/batchnorm/ReadVariableOp_2¢4batch_normalization_314/batchnorm/mul/ReadVariableOp¢!conv2d_440/BiasAdd/ReadVariableOp¢ conv2d_440/Conv2D/ReadVariableOp¢!conv2d_441/BiasAdd/ReadVariableOp¢ conv2d_441/Conv2D/ReadVariableOp¢!conv2d_442/BiasAdd/ReadVariableOp¢ conv2d_442/Conv2D/ReadVariableOp¢dense_69/BiasAdd/ReadVariableOp¢dense_69/MatMul/ReadVariableOp¢$gender_output/BiasAdd/ReadVariableOp¢#gender_output/MatMul/ReadVariableOp
 conv2d_440/Conv2D/ReadVariableOpReadVariableOp)conv2d_440_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¯
conv2d_440/Conv2DConv2Dinputs(conv2d_440/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

!conv2d_440/BiasAdd/ReadVariableOpReadVariableOp*conv2d_440_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_440/BiasAddBiasAddconv2d_440/Conv2D:output:0)conv2d_440/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  n
conv2d_440/ReluReluconv2d_440/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
&batch_normalization_311/ReadVariableOpReadVariableOp/batch_normalization_311_readvariableop_resource*
_output_shapes
:*
dtype0
(batch_normalization_311/ReadVariableOp_1ReadVariableOp1batch_normalization_311_readvariableop_1_resource*
_output_shapes
:*
dtype0´
7batch_normalization_311/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_311_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¸
9batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_311_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Å
(batch_normalization_311/FusedBatchNormV3FusedBatchNormV3conv2d_440/Relu:activations:0.batch_normalization_311/ReadVariableOp:value:00batch_normalization_311/ReadVariableOp_1:value:0?batch_normalization_311/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_311/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( ¿
max_pooling2d_114/MaxPoolMaxPool,batch_normalization_311/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

 conv2d_441/Conv2D/ReadVariableOpReadVariableOp)conv2d_441_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ë
conv2d_441/Conv2DConv2D"max_pooling2d_114/MaxPool:output:0(conv2d_441/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

!conv2d_441/BiasAdd/ReadVariableOpReadVariableOp*conv2d_441_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_441/BiasAddBiasAddconv2d_441/Conv2D:output:0)conv2d_441/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
conv2d_441/ReluReluconv2d_441/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&batch_normalization_312/ReadVariableOpReadVariableOp/batch_normalization_312_readvariableop_resource*
_output_shapes
: *
dtype0
(batch_normalization_312/ReadVariableOp_1ReadVariableOp1batch_normalization_312_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7batch_normalization_312/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_312_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_312_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Å
(batch_normalization_312/FusedBatchNormV3FusedBatchNormV3conv2d_441/Relu:activations:0.batch_normalization_312/ReadVariableOp:value:00batch_normalization_312/ReadVariableOp_1:value:0?batch_normalization_312/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_312/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ¿
max_pooling2d_115/MaxPoolMaxPool,batch_normalization_312/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

 conv2d_442/Conv2D/ReadVariableOpReadVariableOp)conv2d_442_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ë
conv2d_442/Conv2DConv2D"max_pooling2d_115/MaxPool:output:0(conv2d_442/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

!conv2d_442/BiasAdd/ReadVariableOpReadVariableOp*conv2d_442_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_442/BiasAddBiasAddconv2d_442/Conv2D:output:0)conv2d_442/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
conv2d_442/ReluReluconv2d_442/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&batch_normalization_313/ReadVariableOpReadVariableOp/batch_normalization_313_readvariableop_resource*
_output_shapes
:@*
dtype0
(batch_normalization_313/ReadVariableOp_1ReadVariableOp1batch_normalization_313_readvariableop_1_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_313/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_313_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¸
9batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_313_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Å
(batch_normalization_313/FusedBatchNormV3FusedBatchNormV3conv2d_442/Relu:activations:0.batch_normalization_313/ReadVariableOp:value:00batch_normalization_313/ReadVariableOp_1:value:0?batch_normalization_313/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_313/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ¿
max_pooling2d_116/MaxPoolMaxPool,batch_normalization_313/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
a
flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_52/ReshapeReshape"max_pooling2d_116/MaxPool:output:0flatten_52/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_69/MatMulMatMulflatten_52/Reshape:output:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_69/ReluReludense_69/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0batch_normalization_314/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_314_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0l
'batch_normalization_314/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:À
%batch_normalization_314/batchnorm/addAddV28batch_normalization_314/batchnorm/ReadVariableOp:value:00batch_normalization_314/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
'batch_normalization_314/batchnorm/RsqrtRsqrt)batch_normalization_314/batchnorm/add:z:0*
T0*
_output_shapes	
:¯
4batch_normalization_314/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_314_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0½
%batch_normalization_314/batchnorm/mulMul+batch_normalization_314/batchnorm/Rsqrt:y:0<batch_normalization_314/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:©
'batch_normalization_314/batchnorm/mul_1Muldense_69/Relu:activations:0)batch_normalization_314/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2batch_normalization_314/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_314_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0»
'batch_normalization_314/batchnorm/mul_2Mul:batch_normalization_314/batchnorm/ReadVariableOp_1:value:0)batch_normalization_314/batchnorm/mul:z:0*
T0*
_output_shapes	
:«
2batch_normalization_314/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_314_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0»
%batch_normalization_314/batchnorm/subSub:batch_normalization_314/batchnorm/ReadVariableOp_2:value:0+batch_normalization_314/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:»
'batch_normalization_314/batchnorm/add_1AddV2+batch_normalization_314/batchnorm/mul_1:z:0)batch_normalization_314/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_72/IdentityIdentity+batch_normalization_314/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#gender_output/MatMul/ReadVariableOpReadVariableOp,gender_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gender_output/MatMulMatMuldropout_72/Identity:output:0+gender_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$gender_output/BiasAdd/ReadVariableOpReadVariableOp-gender_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
gender_output/BiasAddBiasAddgender_output/MatMul:product:0,gender_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gender_output/SoftmaxSoftmaxgender_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
IdentityIdentitygender_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
NoOpNoOp8^batch_normalization_311/FusedBatchNormV3/ReadVariableOp:^batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_311/ReadVariableOp)^batch_normalization_311/ReadVariableOp_18^batch_normalization_312/FusedBatchNormV3/ReadVariableOp:^batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_312/ReadVariableOp)^batch_normalization_312/ReadVariableOp_18^batch_normalization_313/FusedBatchNormV3/ReadVariableOp:^batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_313/ReadVariableOp)^batch_normalization_313/ReadVariableOp_11^batch_normalization_314/batchnorm/ReadVariableOp3^batch_normalization_314/batchnorm/ReadVariableOp_13^batch_normalization_314/batchnorm/ReadVariableOp_25^batch_normalization_314/batchnorm/mul/ReadVariableOp"^conv2d_440/BiasAdd/ReadVariableOp!^conv2d_440/Conv2D/ReadVariableOp"^conv2d_441/BiasAdd/ReadVariableOp!^conv2d_441/Conv2D/ReadVariableOp"^conv2d_442/BiasAdd/ReadVariableOp!^conv2d_442/Conv2D/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp%^gender_output/BiasAdd/ReadVariableOp$^gender_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_311/FusedBatchNormV3/ReadVariableOp7batch_normalization_311/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_311/FusedBatchNormV3/ReadVariableOp_19batch_normalization_311/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_311/ReadVariableOp&batch_normalization_311/ReadVariableOp2T
(batch_normalization_311/ReadVariableOp_1(batch_normalization_311/ReadVariableOp_12r
7batch_normalization_312/FusedBatchNormV3/ReadVariableOp7batch_normalization_312/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_312/FusedBatchNormV3/ReadVariableOp_19batch_normalization_312/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_312/ReadVariableOp&batch_normalization_312/ReadVariableOp2T
(batch_normalization_312/ReadVariableOp_1(batch_normalization_312/ReadVariableOp_12r
7batch_normalization_313/FusedBatchNormV3/ReadVariableOp7batch_normalization_313/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_313/FusedBatchNormV3/ReadVariableOp_19batch_normalization_313/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_313/ReadVariableOp&batch_normalization_313/ReadVariableOp2T
(batch_normalization_313/ReadVariableOp_1(batch_normalization_313/ReadVariableOp_12d
0batch_normalization_314/batchnorm/ReadVariableOp0batch_normalization_314/batchnorm/ReadVariableOp2h
2batch_normalization_314/batchnorm/ReadVariableOp_12batch_normalization_314/batchnorm/ReadVariableOp_12h
2batch_normalization_314/batchnorm/ReadVariableOp_22batch_normalization_314/batchnorm/ReadVariableOp_22l
4batch_normalization_314/batchnorm/mul/ReadVariableOp4batch_normalization_314/batchnorm/mul/ReadVariableOp2F
!conv2d_440/BiasAdd/ReadVariableOp!conv2d_440/BiasAdd/ReadVariableOp2D
 conv2d_440/Conv2D/ReadVariableOp conv2d_440/Conv2D/ReadVariableOp2F
!conv2d_441/BiasAdd/ReadVariableOp!conv2d_441/BiasAdd/ReadVariableOp2D
 conv2d_441/Conv2D/ReadVariableOp conv2d_441/Conv2D/ReadVariableOp2F
!conv2d_442/BiasAdd/ReadVariableOp!conv2d_442/BiasAdd/ReadVariableOp2D
 conv2d_442/Conv2D/ReadVariableOp conv2d_442/Conv2D/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp2L
$gender_output/BiasAdd/ReadVariableOp$gender_output/BiasAdd/ReadVariableOp2J
#gender_output/MatMul/ReadVariableOp#gender_output/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

Ã
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228965

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_313_layer_call_fn_1230368

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228965
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ã
°
*__inference_model_53_layer_call_fn_1229592
input_56!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_56unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_53_layer_call_and_return_conditional_losses_1229480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
input_56
Ã
«
%__inference_signature_wrapper_1229797
input_56!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_56unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1228760o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
input_56
ª

ü
J__inference_gender_output_layer_call_and_return_conditional_losses_1230572

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
Å/
#__inference__traced_restore_1231031
file_prefix<
"assignvariableop_conv2d_440_kernel:0
"assignvariableop_1_conv2d_440_bias:>
0assignvariableop_2_batch_normalization_311_gamma:=
/assignvariableop_3_batch_normalization_311_beta:D
6assignvariableop_4_batch_normalization_311_moving_mean:H
:assignvariableop_5_batch_normalization_311_moving_variance:>
$assignvariableop_6_conv2d_441_kernel: 0
"assignvariableop_7_conv2d_441_bias: >
0assignvariableop_8_batch_normalization_312_gamma: =
/assignvariableop_9_batch_normalization_312_beta: E
7assignvariableop_10_batch_normalization_312_moving_mean: I
;assignvariableop_11_batch_normalization_312_moving_variance: ?
%assignvariableop_12_conv2d_442_kernel: @1
#assignvariableop_13_conv2d_442_bias:@?
1assignvariableop_14_batch_normalization_313_gamma:@>
0assignvariableop_15_batch_normalization_313_beta:@E
7assignvariableop_16_batch_normalization_313_moving_mean:@I
;assignvariableop_17_batch_normalization_313_moving_variance:@7
#assignvariableop_18_dense_69_kernel:
0
!assignvariableop_19_dense_69_bias:	@
1assignvariableop_20_batch_normalization_314_gamma:	?
0assignvariableop_21_batch_normalization_314_beta:	F
7assignvariableop_22_batch_normalization_314_moving_mean:	J
;assignvariableop_23_batch_normalization_314_moving_variance:	;
(assignvariableop_24_gender_output_kernel:	4
&assignvariableop_25_gender_output_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: #
assignvariableop_33_total: #
assignvariableop_34_count: F
,assignvariableop_35_adam_conv2d_440_kernel_m:8
*assignvariableop_36_adam_conv2d_440_bias_m:F
8assignvariableop_37_adam_batch_normalization_311_gamma_m:E
7assignvariableop_38_adam_batch_normalization_311_beta_m:F
,assignvariableop_39_adam_conv2d_441_kernel_m: 8
*assignvariableop_40_adam_conv2d_441_bias_m: F
8assignvariableop_41_adam_batch_normalization_312_gamma_m: E
7assignvariableop_42_adam_batch_normalization_312_beta_m: F
,assignvariableop_43_adam_conv2d_442_kernel_m: @8
*assignvariableop_44_adam_conv2d_442_bias_m:@F
8assignvariableop_45_adam_batch_normalization_313_gamma_m:@E
7assignvariableop_46_adam_batch_normalization_313_beta_m:@>
*assignvariableop_47_adam_dense_69_kernel_m:
7
(assignvariableop_48_adam_dense_69_bias_m:	G
8assignvariableop_49_adam_batch_normalization_314_gamma_m:	F
7assignvariableop_50_adam_batch_normalization_314_beta_m:	B
/assignvariableop_51_adam_gender_output_kernel_m:	;
-assignvariableop_52_adam_gender_output_bias_m:F
,assignvariableop_53_adam_conv2d_440_kernel_v:8
*assignvariableop_54_adam_conv2d_440_bias_v:F
8assignvariableop_55_adam_batch_normalization_311_gamma_v:E
7assignvariableop_56_adam_batch_normalization_311_beta_v:F
,assignvariableop_57_adam_conv2d_441_kernel_v: 8
*assignvariableop_58_adam_conv2d_441_bias_v: F
8assignvariableop_59_adam_batch_normalization_312_gamma_v: E
7assignvariableop_60_adam_batch_normalization_312_beta_v: F
,assignvariableop_61_adam_conv2d_442_kernel_v: @8
*assignvariableop_62_adam_conv2d_442_bias_v:@F
8assignvariableop_63_adam_batch_normalization_313_gamma_v:@E
7assignvariableop_64_adam_batch_normalization_313_beta_v:@>
*assignvariableop_65_adam_dense_69_kernel_v:
7
(assignvariableop_66_adam_dense_69_bias_v:	G
8assignvariableop_67_adam_batch_normalization_314_gamma_v:	F
7assignvariableop_68_adam_batch_normalization_314_beta_v:	B
/assignvariableop_69_adam_gender_output_kernel_v:	;
-assignvariableop_70_adam_gender_output_bias_v:
identity_72¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_8¢AssignVariableOp_9À'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*æ&
valueÜ&BÙ&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*¥
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_440_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_440_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_311_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_311_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_311_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_311_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_441_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_441_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_312_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_312_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_312_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_312_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_442_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_442_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batch_normalization_313_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_313_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batch_normalization_313_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batch_normalization_313_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_69_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_69_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_20AssignVariableOp1assignvariableop_20_batch_normalization_314_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_314_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_314_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_23AssignVariableOp;assignvariableop_23_batch_normalization_314_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_gender_output_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp&assignvariableop_25_gender_output_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_440_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_440_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_37AssignVariableOp8assignvariableop_37_adam_batch_normalization_311_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adam_batch_normalization_311_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_441_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_441_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_41AssignVariableOp8assignvariableop_41_adam_batch_normalization_312_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_batch_normalization_312_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_442_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_442_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_45AssignVariableOp8assignvariableop_45_adam_batch_normalization_313_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_batch_normalization_313_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_69_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_69_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_49AssignVariableOp8assignvariableop_49_adam_batch_normalization_314_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_314_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_51AssignVariableOp/assignvariableop_51_adam_gender_output_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp-assignvariableop_52_adam_gender_output_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_440_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_440_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_55AssignVariableOp8assignvariableop_55_adam_batch_normalization_311_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_batch_normalization_311_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_441_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_441_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_batch_normalization_312_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_312_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_442_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_442_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_63AssignVariableOp8assignvariableop_63_adam_batch_normalization_313_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_batch_normalization_313_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_69_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_69_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_batch_normalization_314_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_314_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_69AssignVariableOp/assignvariableop_69_adam_gender_output_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp-assignvariableop_70_adam_gender_output_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 é
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_72IdentityIdentity_71:output:0^NoOp_1*
T0*
_output_shapes
: Ö
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_72Identity_72:output:0*¥
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
K

E__inference_model_53_layer_call_and_return_conditional_losses_1229480

inputs,
conv2d_440_1229413: 
conv2d_440_1229415:-
batch_normalization_311_1229418:-
batch_normalization_311_1229420:-
batch_normalization_311_1229422:-
batch_normalization_311_1229424:,
conv2d_441_1229428:  
conv2d_441_1229430: -
batch_normalization_312_1229433: -
batch_normalization_312_1229435: -
batch_normalization_312_1229437: -
batch_normalization_312_1229439: ,
conv2d_442_1229443: @ 
conv2d_442_1229445:@-
batch_normalization_313_1229448:@-
batch_normalization_313_1229450:@-
batch_normalization_313_1229452:@-
batch_normalization_313_1229454:@$
dense_69_1229459:

dense_69_1229461:	.
batch_normalization_314_1229464:	.
batch_normalization_314_1229466:	.
batch_normalization_314_1229468:	.
batch_normalization_314_1229470:	(
gender_output_1229474:	#
gender_output_1229476:
identity¢/batch_normalization_311/StatefulPartitionedCall¢/batch_normalization_312/StatefulPartitionedCall¢/batch_normalization_313/StatefulPartitionedCall¢/batch_normalization_314/StatefulPartitionedCall¢"conv2d_440/StatefulPartitionedCall¢"conv2d_441/StatefulPartitionedCall¢"conv2d_442/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢"dropout_72/StatefulPartitionedCall¢%gender_output/StatefulPartitionedCall
"conv2d_440/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_440_1229413conv2d_440_1229415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1229088£
/batch_normalization_311/StatefulPartitionedCallStatefulPartitionedCall+conv2d_440/StatefulPartitionedCall:output:0batch_normalization_311_1229418batch_normalization_311_1229420batch_normalization_311_1229422batch_normalization_311_1229424*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228813
!max_pooling2d_114/PartitionedCallPartitionedCall8batch_normalization_311/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1228833ª
"conv2d_441/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_114/PartitionedCall:output:0conv2d_441_1229428conv2d_441_1229430*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1229115£
/batch_normalization_312/StatefulPartitionedCallStatefulPartitionedCall+conv2d_441/StatefulPartitionedCall:output:0batch_normalization_312_1229433batch_normalization_312_1229435batch_normalization_312_1229437batch_normalization_312_1229439*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228889
!max_pooling2d_115/PartitionedCallPartitionedCall8batch_normalization_312/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1228909ª
"conv2d_442/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_115/PartitionedCall:output:0conv2d_442_1229443conv2d_442_1229445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1229142£
/batch_normalization_313/StatefulPartitionedCallStatefulPartitionedCall+conv2d_442/StatefulPartitionedCall:output:0batch_normalization_313_1229448batch_normalization_313_1229450batch_normalization_313_1229452batch_normalization_313_1229454*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228965
!max_pooling2d_116/PartitionedCallPartitionedCall8batch_normalization_313/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1228985å
flatten_52/PartitionedCallPartitionedCall*max_pooling2d_116/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_52_layer_call_and_return_conditional_losses_1229164
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_69_1229459dense_69_1229461*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_1229177
/batch_normalization_314/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0batch_normalization_314_1229464batch_normalization_314_1229466batch_normalization_314_1229468batch_normalization_314_1229470*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229059
"dropout_72/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_314/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229302¯
%gender_output/StatefulPartitionedCallStatefulPartitionedCall+dropout_72/StatefulPartitionedCall:output:0gender_output_1229474gender_output_1229476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gender_output_layer_call_and_return_conditional_losses_1229210}
IdentityIdentity.gender_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp0^batch_normalization_311/StatefulPartitionedCall0^batch_normalization_312/StatefulPartitionedCall0^batch_normalization_313/StatefulPartitionedCall0^batch_normalization_314/StatefulPartitionedCall#^conv2d_440/StatefulPartitionedCall#^conv2d_441/StatefulPartitionedCall#^conv2d_442/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall#^dropout_72/StatefulPartitionedCall&^gender_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_311/StatefulPartitionedCall/batch_normalization_311/StatefulPartitionedCall2b
/batch_normalization_312/StatefulPartitionedCall/batch_normalization_312/StatefulPartitionedCall2b
/batch_normalization_313/StatefulPartitionedCall/batch_normalization_313/StatefulPartitionedCall2b
/batch_normalization_314/StatefulPartitionedCall/batch_normalization_314/StatefulPartitionedCall2H
"conv2d_440/StatefulPartitionedCall"conv2d_440/StatefulPartitionedCall2H
"conv2d_441/StatefulPartitionedCall"conv2d_441/StatefulPartitionedCall2H
"conv2d_442/StatefulPartitionedCall"conv2d_442/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2H
"dropout_72/StatefulPartitionedCall"dropout_72/StatefulPartitionedCall2N
%gender_output/StatefulPartitionedCall%gender_output/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ª
H
,__inference_dropout_72_layer_call_fn_1230530

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229197a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

*__inference_dense_69_layer_call_fn_1230434

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_1229177p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

ü
J__inference_gender_output_layer_call_and_return_conditional_losses_1229210

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_conv2d_440_layer_call_and_return_conditional_losses_1230158

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ý	
f
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230552

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
K

E__inference_model_53_layer_call_and_return_conditional_losses_1229732
input_56,
conv2d_440_1229665: 
conv2d_440_1229667:-
batch_normalization_311_1229670:-
batch_normalization_311_1229672:-
batch_normalization_311_1229674:-
batch_normalization_311_1229676:,
conv2d_441_1229680:  
conv2d_441_1229682: -
batch_normalization_312_1229685: -
batch_normalization_312_1229687: -
batch_normalization_312_1229689: -
batch_normalization_312_1229691: ,
conv2d_442_1229695: @ 
conv2d_442_1229697:@-
batch_normalization_313_1229700:@-
batch_normalization_313_1229702:@-
batch_normalization_313_1229704:@-
batch_normalization_313_1229706:@$
dense_69_1229711:

dense_69_1229713:	.
batch_normalization_314_1229716:	.
batch_normalization_314_1229718:	.
batch_normalization_314_1229720:	.
batch_normalization_314_1229722:	(
gender_output_1229726:	#
gender_output_1229728:
identity¢/batch_normalization_311/StatefulPartitionedCall¢/batch_normalization_312/StatefulPartitionedCall¢/batch_normalization_313/StatefulPartitionedCall¢/batch_normalization_314/StatefulPartitionedCall¢"conv2d_440/StatefulPartitionedCall¢"conv2d_441/StatefulPartitionedCall¢"conv2d_442/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢"dropout_72/StatefulPartitionedCall¢%gender_output/StatefulPartitionedCall
"conv2d_440/StatefulPartitionedCallStatefulPartitionedCallinput_56conv2d_440_1229665conv2d_440_1229667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1229088£
/batch_normalization_311/StatefulPartitionedCallStatefulPartitionedCall+conv2d_440/StatefulPartitionedCall:output:0batch_normalization_311_1229670batch_normalization_311_1229672batch_normalization_311_1229674batch_normalization_311_1229676*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228813
!max_pooling2d_114/PartitionedCallPartitionedCall8batch_normalization_311/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1228833ª
"conv2d_441/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_114/PartitionedCall:output:0conv2d_441_1229680conv2d_441_1229682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1229115£
/batch_normalization_312/StatefulPartitionedCallStatefulPartitionedCall+conv2d_441/StatefulPartitionedCall:output:0batch_normalization_312_1229685batch_normalization_312_1229687batch_normalization_312_1229689batch_normalization_312_1229691*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228889
!max_pooling2d_115/PartitionedCallPartitionedCall8batch_normalization_312/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1228909ª
"conv2d_442/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_115/PartitionedCall:output:0conv2d_442_1229695conv2d_442_1229697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1229142£
/batch_normalization_313/StatefulPartitionedCallStatefulPartitionedCall+conv2d_442/StatefulPartitionedCall:output:0batch_normalization_313_1229700batch_normalization_313_1229702batch_normalization_313_1229704batch_normalization_313_1229706*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228965
!max_pooling2d_116/PartitionedCallPartitionedCall8batch_normalization_313/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1228985å
flatten_52/PartitionedCallPartitionedCall*max_pooling2d_116/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_52_layer_call_and_return_conditional_losses_1229164
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_69_1229711dense_69_1229713*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_1229177
/batch_normalization_314/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0batch_normalization_314_1229716batch_normalization_314_1229718batch_normalization_314_1229720batch_normalization_314_1229722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229059
"dropout_72/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_314/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229302¯
%gender_output/StatefulPartitionedCallStatefulPartitionedCall+dropout_72/StatefulPartitionedCall:output:0gender_output_1229726gender_output_1229728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gender_output_layer_call_and_return_conditional_losses_1229210}
IdentityIdentity.gender_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp0^batch_normalization_311/StatefulPartitionedCall0^batch_normalization_312/StatefulPartitionedCall0^batch_normalization_313/StatefulPartitionedCall0^batch_normalization_314/StatefulPartitionedCall#^conv2d_440/StatefulPartitionedCall#^conv2d_441/StatefulPartitionedCall#^conv2d_442/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall#^dropout_72/StatefulPartitionedCall&^gender_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_311/StatefulPartitionedCall/batch_normalization_311/StatefulPartitionedCall2b
/batch_normalization_312/StatefulPartitionedCall/batch_normalization_312/StatefulPartitionedCall2b
/batch_normalization_313/StatefulPartitionedCall/batch_normalization_313/StatefulPartitionedCall2b
/batch_normalization_314/StatefulPartitionedCall/batch_normalization_314/StatefulPartitionedCall2H
"conv2d_440/StatefulPartitionedCall"conv2d_440/StatefulPartitionedCall2H
"conv2d_441/StatefulPartitionedCall"conv2d_441/StatefulPartitionedCall2H
"conv2d_442/StatefulPartitionedCall"conv2d_442/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2H
"dropout_72/StatefulPartitionedCall"dropout_72/StatefulPartitionedCall2N
%gender_output/StatefulPartitionedCall%gender_output/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
input_56
¨

ù
E__inference_dense_69_layer_call_and_return_conditional_losses_1230445

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228782

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
°
*__inference_model_53_layer_call_fn_1229272
input_56!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinput_56unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_53_layer_call_and_return_conditional_losses_1229217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
input_56

Ã
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228889

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó
¡
,__inference_conv2d_442_layer_call_fn_1230331

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1229142w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


G__inference_conv2d_441_layer_call_and_return_conditional_losses_1230250

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
¡
,__inference_conv2d_440_layer_call_fn_1230147

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1229088w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs


G__inference_conv2d_442_layer_call_and_return_conditional_losses_1229142

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶¾
è
E__inference_model_53_layer_call_and_return_conditional_losses_1230138

inputsC
)conv2d_440_conv2d_readvariableop_resource:8
*conv2d_440_biasadd_readvariableop_resource:=
/batch_normalization_311_readvariableop_resource:?
1batch_normalization_311_readvariableop_1_resource:N
@batch_normalization_311_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_311_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_441_conv2d_readvariableop_resource: 8
*conv2d_441_biasadd_readvariableop_resource: =
/batch_normalization_312_readvariableop_resource: ?
1batch_normalization_312_readvariableop_1_resource: N
@batch_normalization_312_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_312_fusedbatchnormv3_readvariableop_1_resource: C
)conv2d_442_conv2d_readvariableop_resource: @8
*conv2d_442_biasadd_readvariableop_resource:@=
/batch_normalization_313_readvariableop_resource:@?
1batch_normalization_313_readvariableop_1_resource:@N
@batch_normalization_313_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_313_fusedbatchnormv3_readvariableop_1_resource:@;
'dense_69_matmul_readvariableop_resource:
7
(dense_69_biasadd_readvariableop_resource:	N
?batch_normalization_314_assignmovingavg_readvariableop_resource:	P
Abatch_normalization_314_assignmovingavg_1_readvariableop_resource:	L
=batch_normalization_314_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_314_batchnorm_readvariableop_resource:	?
,gender_output_matmul_readvariableop_resource:	;
-gender_output_biasadd_readvariableop_resource:
identity¢&batch_normalization_311/AssignNewValue¢(batch_normalization_311/AssignNewValue_1¢7batch_normalization_311/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_311/ReadVariableOp¢(batch_normalization_311/ReadVariableOp_1¢&batch_normalization_312/AssignNewValue¢(batch_normalization_312/AssignNewValue_1¢7batch_normalization_312/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_312/ReadVariableOp¢(batch_normalization_312/ReadVariableOp_1¢&batch_normalization_313/AssignNewValue¢(batch_normalization_313/AssignNewValue_1¢7batch_normalization_313/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_313/ReadVariableOp¢(batch_normalization_313/ReadVariableOp_1¢'batch_normalization_314/AssignMovingAvg¢6batch_normalization_314/AssignMovingAvg/ReadVariableOp¢)batch_normalization_314/AssignMovingAvg_1¢8batch_normalization_314/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_314/batchnorm/ReadVariableOp¢4batch_normalization_314/batchnorm/mul/ReadVariableOp¢!conv2d_440/BiasAdd/ReadVariableOp¢ conv2d_440/Conv2D/ReadVariableOp¢!conv2d_441/BiasAdd/ReadVariableOp¢ conv2d_441/Conv2D/ReadVariableOp¢!conv2d_442/BiasAdd/ReadVariableOp¢ conv2d_442/Conv2D/ReadVariableOp¢dense_69/BiasAdd/ReadVariableOp¢dense_69/MatMul/ReadVariableOp¢$gender_output/BiasAdd/ReadVariableOp¢#gender_output/MatMul/ReadVariableOp
 conv2d_440/Conv2D/ReadVariableOpReadVariableOp)conv2d_440_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¯
conv2d_440/Conv2DConv2Dinputs(conv2d_440/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

!conv2d_440/BiasAdd/ReadVariableOpReadVariableOp*conv2d_440_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_440/BiasAddBiasAddconv2d_440/Conv2D:output:0)conv2d_440/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  n
conv2d_440/ReluReluconv2d_440/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
&batch_normalization_311/ReadVariableOpReadVariableOp/batch_normalization_311_readvariableop_resource*
_output_shapes
:*
dtype0
(batch_normalization_311/ReadVariableOp_1ReadVariableOp1batch_normalization_311_readvariableop_1_resource*
_output_shapes
:*
dtype0´
7batch_normalization_311/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_311_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¸
9batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_311_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ó
(batch_normalization_311/FusedBatchNormV3FusedBatchNormV3conv2d_440/Relu:activations:0.batch_normalization_311/ReadVariableOp:value:00batch_normalization_311/ReadVariableOp_1:value:0?batch_normalization_311/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_311/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<¦
&batch_normalization_311/AssignNewValueAssignVariableOp@batch_normalization_311_fusedbatchnormv3_readvariableop_resource5batch_normalization_311/FusedBatchNormV3:batch_mean:08^batch_normalization_311/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
(batch_normalization_311/AssignNewValue_1AssignVariableOpBbatch_normalization_311_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_311/FusedBatchNormV3:batch_variance:0:^batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(¿
max_pooling2d_114/MaxPoolMaxPool,batch_normalization_311/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

 conv2d_441/Conv2D/ReadVariableOpReadVariableOp)conv2d_441_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ë
conv2d_441/Conv2DConv2D"max_pooling2d_114/MaxPool:output:0(conv2d_441/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

!conv2d_441/BiasAdd/ReadVariableOpReadVariableOp*conv2d_441_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_441/BiasAddBiasAddconv2d_441/Conv2D:output:0)conv2d_441/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
conv2d_441/ReluReluconv2d_441/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&batch_normalization_312/ReadVariableOpReadVariableOp/batch_normalization_312_readvariableop_resource*
_output_shapes
: *
dtype0
(batch_normalization_312/ReadVariableOp_1ReadVariableOp1batch_normalization_312_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7batch_normalization_312/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_312_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_312_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ó
(batch_normalization_312/FusedBatchNormV3FusedBatchNormV3conv2d_441/Relu:activations:0.batch_normalization_312/ReadVariableOp:value:00batch_normalization_312/ReadVariableOp_1:value:0?batch_normalization_312/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_312/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<¦
&batch_normalization_312/AssignNewValueAssignVariableOp@batch_normalization_312_fusedbatchnormv3_readvariableop_resource5batch_normalization_312/FusedBatchNormV3:batch_mean:08^batch_normalization_312/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
(batch_normalization_312/AssignNewValue_1AssignVariableOpBbatch_normalization_312_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_312/FusedBatchNormV3:batch_variance:0:^batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(¿
max_pooling2d_115/MaxPoolMaxPool,batch_normalization_312/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

 conv2d_442/Conv2D/ReadVariableOpReadVariableOp)conv2d_442_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ë
conv2d_442/Conv2DConv2D"max_pooling2d_115/MaxPool:output:0(conv2d_442/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

!conv2d_442/BiasAdd/ReadVariableOpReadVariableOp*conv2d_442_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_442/BiasAddBiasAddconv2d_442/Conv2D:output:0)conv2d_442/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
conv2d_442/ReluReluconv2d_442/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&batch_normalization_313/ReadVariableOpReadVariableOp/batch_normalization_313_readvariableop_resource*
_output_shapes
:@*
dtype0
(batch_normalization_313/ReadVariableOp_1ReadVariableOp1batch_normalization_313_readvariableop_1_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_313/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_313_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¸
9batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_313_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ó
(batch_normalization_313/FusedBatchNormV3FusedBatchNormV3conv2d_442/Relu:activations:0.batch_normalization_313/ReadVariableOp:value:00batch_normalization_313/ReadVariableOp_1:value:0?batch_normalization_313/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_313/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<¦
&batch_normalization_313/AssignNewValueAssignVariableOp@batch_normalization_313_fusedbatchnormv3_readvariableop_resource5batch_normalization_313/FusedBatchNormV3:batch_mean:08^batch_normalization_313/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
(batch_normalization_313/AssignNewValue_1AssignVariableOpBbatch_normalization_313_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_313/FusedBatchNormV3:batch_variance:0:^batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(¿
max_pooling2d_116/MaxPoolMaxPool,batch_normalization_313/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
a
flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_52/ReshapeReshape"max_pooling2d_116/MaxPool:output:0flatten_52/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_69/MatMulMatMulflatten_52/Reshape:output:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_69/ReluReludense_69/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_314/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Å
$batch_normalization_314/moments/meanMeandense_69/Relu:activations:0?batch_normalization_314/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
,batch_normalization_314/moments/StopGradientStopGradient-batch_normalization_314/moments/mean:output:0*
T0*
_output_shapes
:	Í
1batch_normalization_314/moments/SquaredDifferenceSquaredDifferencedense_69/Relu:activations:05batch_normalization_314/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_314/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ç
(batch_normalization_314/moments/varianceMean5batch_normalization_314/moments/SquaredDifference:z:0Cbatch_normalization_314/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
'batch_normalization_314/moments/SqueezeSqueeze-batch_normalization_314/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¤
)batch_normalization_314/moments/Squeeze_1Squeeze1batch_normalization_314/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 r
-batch_normalization_314/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<³
6batch_normalization_314/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_314_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ê
+batch_normalization_314/AssignMovingAvg/subSub>batch_normalization_314/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_314/moments/Squeeze:output:0*
T0*
_output_shapes	
:Á
+batch_normalization_314/AssignMovingAvg/mulMul/batch_normalization_314/AssignMovingAvg/sub:z:06batch_normalization_314/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
'batch_normalization_314/AssignMovingAvgAssignSubVariableOp?batch_normalization_314_assignmovingavg_readvariableop_resource/batch_normalization_314/AssignMovingAvg/mul:z:07^batch_normalization_314/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_314/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<·
8batch_normalization_314/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_314_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ð
-batch_normalization_314/AssignMovingAvg_1/subSub@batch_normalization_314/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_314/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ç
-batch_normalization_314/AssignMovingAvg_1/mulMul1batch_normalization_314/AssignMovingAvg_1/sub:z:08batch_normalization_314/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
)batch_normalization_314/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_314_assignmovingavg_1_readvariableop_resource1batch_normalization_314/AssignMovingAvg_1/mul:z:09^batch_normalization_314/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_314/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
%batch_normalization_314/batchnorm/addAddV22batch_normalization_314/moments/Squeeze_1:output:00batch_normalization_314/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
'batch_normalization_314/batchnorm/RsqrtRsqrt)batch_normalization_314/batchnorm/add:z:0*
T0*
_output_shapes	
:¯
4batch_normalization_314/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_314_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0½
%batch_normalization_314/batchnorm/mulMul+batch_normalization_314/batchnorm/Rsqrt:y:0<batch_normalization_314/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:©
'batch_normalization_314/batchnorm/mul_1Muldense_69/Relu:activations:0)batch_normalization_314/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
'batch_normalization_314/batchnorm/mul_2Mul0batch_normalization_314/moments/Squeeze:output:0)batch_normalization_314/batchnorm/mul:z:0*
T0*
_output_shapes	
:§
0batch_normalization_314/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_314_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¹
%batch_normalization_314/batchnorm/subSub8batch_normalization_314/batchnorm/ReadVariableOp:value:0+batch_normalization_314/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:»
'batch_normalization_314/batchnorm/add_1AddV2+batch_normalization_314/batchnorm/mul_1:z:0)batch_normalization_314/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_72/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ? 
dropout_72/dropout/MulMul+batch_normalization_314/batchnorm/add_1:z:0!dropout_72/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_72/dropout/ShapeShape+batch_normalization_314/batchnorm/add_1:z:0*
T0*
_output_shapes
:£
/dropout_72/dropout/random_uniform/RandomUniformRandomUniform!dropout_72/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_72/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>È
dropout_72/dropout/GreaterEqualGreaterEqual8dropout_72/dropout/random_uniform/RandomUniform:output:0*dropout_72/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_72/dropout/CastCast#dropout_72/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_72/dropout/Mul_1Muldropout_72/dropout/Mul:z:0dropout_72/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#gender_output/MatMul/ReadVariableOpReadVariableOp,gender_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gender_output/MatMulMatMuldropout_72/dropout/Mul_1:z:0+gender_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$gender_output/BiasAdd/ReadVariableOpReadVariableOp-gender_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
gender_output/BiasAddBiasAddgender_output/MatMul:product:0,gender_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gender_output/SoftmaxSoftmaxgender_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
IdentityIdentitygender_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
NoOpNoOp'^batch_normalization_311/AssignNewValue)^batch_normalization_311/AssignNewValue_18^batch_normalization_311/FusedBatchNormV3/ReadVariableOp:^batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_311/ReadVariableOp)^batch_normalization_311/ReadVariableOp_1'^batch_normalization_312/AssignNewValue)^batch_normalization_312/AssignNewValue_18^batch_normalization_312/FusedBatchNormV3/ReadVariableOp:^batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_312/ReadVariableOp)^batch_normalization_312/ReadVariableOp_1'^batch_normalization_313/AssignNewValue)^batch_normalization_313/AssignNewValue_18^batch_normalization_313/FusedBatchNormV3/ReadVariableOp:^batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_313/ReadVariableOp)^batch_normalization_313/ReadVariableOp_1(^batch_normalization_314/AssignMovingAvg7^batch_normalization_314/AssignMovingAvg/ReadVariableOp*^batch_normalization_314/AssignMovingAvg_19^batch_normalization_314/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_314/batchnorm/ReadVariableOp5^batch_normalization_314/batchnorm/mul/ReadVariableOp"^conv2d_440/BiasAdd/ReadVariableOp!^conv2d_440/Conv2D/ReadVariableOp"^conv2d_441/BiasAdd/ReadVariableOp!^conv2d_441/Conv2D/ReadVariableOp"^conv2d_442/BiasAdd/ReadVariableOp!^conv2d_442/Conv2D/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp%^gender_output/BiasAdd/ReadVariableOp$^gender_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_311/AssignNewValue&batch_normalization_311/AssignNewValue2T
(batch_normalization_311/AssignNewValue_1(batch_normalization_311/AssignNewValue_12r
7batch_normalization_311/FusedBatchNormV3/ReadVariableOp7batch_normalization_311/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_311/FusedBatchNormV3/ReadVariableOp_19batch_normalization_311/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_311/ReadVariableOp&batch_normalization_311/ReadVariableOp2T
(batch_normalization_311/ReadVariableOp_1(batch_normalization_311/ReadVariableOp_12P
&batch_normalization_312/AssignNewValue&batch_normalization_312/AssignNewValue2T
(batch_normalization_312/AssignNewValue_1(batch_normalization_312/AssignNewValue_12r
7batch_normalization_312/FusedBatchNormV3/ReadVariableOp7batch_normalization_312/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_312/FusedBatchNormV3/ReadVariableOp_19batch_normalization_312/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_312/ReadVariableOp&batch_normalization_312/ReadVariableOp2T
(batch_normalization_312/ReadVariableOp_1(batch_normalization_312/ReadVariableOp_12P
&batch_normalization_313/AssignNewValue&batch_normalization_313/AssignNewValue2T
(batch_normalization_313/AssignNewValue_1(batch_normalization_313/AssignNewValue_12r
7batch_normalization_313/FusedBatchNormV3/ReadVariableOp7batch_normalization_313/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_313/FusedBatchNormV3/ReadVariableOp_19batch_normalization_313/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_313/ReadVariableOp&batch_normalization_313/ReadVariableOp2T
(batch_normalization_313/ReadVariableOp_1(batch_normalization_313/ReadVariableOp_12R
'batch_normalization_314/AssignMovingAvg'batch_normalization_314/AssignMovingAvg2p
6batch_normalization_314/AssignMovingAvg/ReadVariableOp6batch_normalization_314/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_314/AssignMovingAvg_1)batch_normalization_314/AssignMovingAvg_12t
8batch_normalization_314/AssignMovingAvg_1/ReadVariableOp8batch_normalization_314/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_314/batchnorm/ReadVariableOp0batch_normalization_314/batchnorm/ReadVariableOp2l
4batch_normalization_314/batchnorm/mul/ReadVariableOp4batch_normalization_314/batchnorm/mul/ReadVariableOp2F
!conv2d_440/BiasAdd/ReadVariableOp!conv2d_440/BiasAdd/ReadVariableOp2D
 conv2d_440/Conv2D/ReadVariableOp conv2d_440/Conv2D/ReadVariableOp2F
!conv2d_441/BiasAdd/ReadVariableOp!conv2d_441/BiasAdd/ReadVariableOp2D
 conv2d_441/Conv2D/ReadVariableOp conv2d_441/Conv2D/ReadVariableOp2F
!conv2d_442/BiasAdd/ReadVariableOp!conv2d_442/BiasAdd/ReadVariableOp2D
 conv2d_442/Conv2D/ReadVariableOp conv2d_442/Conv2D/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp2L
$gender_output/BiasAdd/ReadVariableOp$gender_output/BiasAdd/ReadVariableOp2J
#gender_output/MatMul/ReadVariableOp#gender_output/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¨

ù
E__inference_dense_69_layer_call_and_return_conditional_losses_1229177

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228934

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1228909

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1230230

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
·
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230491

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
e
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230540

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
O
3__inference_max_pooling2d_116_layer_call_fn_1230409

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1228985
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
¡
,__inference_conv2d_441_layer_call_fn_1230239

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1229115w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230202

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230386

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñI
à
E__inference_model_53_layer_call_and_return_conditional_losses_1229217

inputs,
conv2d_440_1229089: 
conv2d_440_1229091:-
batch_normalization_311_1229094:-
batch_normalization_311_1229096:-
batch_normalization_311_1229098:-
batch_normalization_311_1229100:,
conv2d_441_1229116:  
conv2d_441_1229118: -
batch_normalization_312_1229121: -
batch_normalization_312_1229123: -
batch_normalization_312_1229125: -
batch_normalization_312_1229127: ,
conv2d_442_1229143: @ 
conv2d_442_1229145:@-
batch_normalization_313_1229148:@-
batch_normalization_313_1229150:@-
batch_normalization_313_1229152:@-
batch_normalization_313_1229154:@$
dense_69_1229178:

dense_69_1229180:	.
batch_normalization_314_1229183:	.
batch_normalization_314_1229185:	.
batch_normalization_314_1229187:	.
batch_normalization_314_1229189:	(
gender_output_1229211:	#
gender_output_1229213:
identity¢/batch_normalization_311/StatefulPartitionedCall¢/batch_normalization_312/StatefulPartitionedCall¢/batch_normalization_313/StatefulPartitionedCall¢/batch_normalization_314/StatefulPartitionedCall¢"conv2d_440/StatefulPartitionedCall¢"conv2d_441/StatefulPartitionedCall¢"conv2d_442/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢%gender_output/StatefulPartitionedCall
"conv2d_440/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_440_1229089conv2d_440_1229091*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1229088¥
/batch_normalization_311/StatefulPartitionedCallStatefulPartitionedCall+conv2d_440/StatefulPartitionedCall:output:0batch_normalization_311_1229094batch_normalization_311_1229096batch_normalization_311_1229098batch_normalization_311_1229100*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228782
!max_pooling2d_114/PartitionedCallPartitionedCall8batch_normalization_311/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1228833ª
"conv2d_441/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_114/PartitionedCall:output:0conv2d_441_1229116conv2d_441_1229118*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1229115¥
/batch_normalization_312/StatefulPartitionedCallStatefulPartitionedCall+conv2d_441/StatefulPartitionedCall:output:0batch_normalization_312_1229121batch_normalization_312_1229123batch_normalization_312_1229125batch_normalization_312_1229127*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228858
!max_pooling2d_115/PartitionedCallPartitionedCall8batch_normalization_312/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1228909ª
"conv2d_442/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_115/PartitionedCall:output:0conv2d_442_1229143conv2d_442_1229145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1229142¥
/batch_normalization_313/StatefulPartitionedCallStatefulPartitionedCall+conv2d_442/StatefulPartitionedCall:output:0batch_normalization_313_1229148batch_normalization_313_1229150batch_normalization_313_1229152batch_normalization_313_1229154*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228934
!max_pooling2d_116/PartitionedCallPartitionedCall8batch_normalization_313/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1228985å
flatten_52/PartitionedCallPartitionedCall*max_pooling2d_116/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_52_layer_call_and_return_conditional_losses_1229164
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_69_1229178dense_69_1229180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_1229177
/batch_normalization_314/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0batch_normalization_314_1229183batch_normalization_314_1229185batch_normalization_314_1229187batch_normalization_314_1229189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229012ó
dropout_72/PartitionedCallPartitionedCall8batch_normalization_314/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229197§
%gender_output/StatefulPartitionedCallStatefulPartitionedCall#dropout_72/PartitionedCall:output:0gender_output_1229211gender_output_1229213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gender_output_layer_call_and_return_conditional_losses_1229210}
IdentityIdentity.gender_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp0^batch_normalization_311/StatefulPartitionedCall0^batch_normalization_312/StatefulPartitionedCall0^batch_normalization_313/StatefulPartitionedCall0^batch_normalization_314/StatefulPartitionedCall#^conv2d_440/StatefulPartitionedCall#^conv2d_441/StatefulPartitionedCall#^conv2d_442/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall&^gender_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_311/StatefulPartitionedCall/batch_normalization_311/StatefulPartitionedCall2b
/batch_normalization_312/StatefulPartitionedCall/batch_normalization_312/StatefulPartitionedCall2b
/batch_normalization_313/StatefulPartitionedCall/batch_normalization_313/StatefulPartitionedCall2b
/batch_normalization_314/StatefulPartitionedCall/batch_normalization_314/StatefulPartitionedCall2H
"conv2d_440/StatefulPartitionedCall"conv2d_440/StatefulPartitionedCall2H
"conv2d_441/StatefulPartitionedCall"conv2d_441/StatefulPartitionedCall2H
"conv2d_442/StatefulPartitionedCall"conv2d_442/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2N
%gender_output/StatefulPartitionedCall%gender_output/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
æ
!
 __inference__traced_save_1230808
file_prefix0
,savev2_conv2d_440_kernel_read_readvariableop.
*savev2_conv2d_440_bias_read_readvariableop<
8savev2_batch_normalization_311_gamma_read_readvariableop;
7savev2_batch_normalization_311_beta_read_readvariableopB
>savev2_batch_normalization_311_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_311_moving_variance_read_readvariableop0
,savev2_conv2d_441_kernel_read_readvariableop.
*savev2_conv2d_441_bias_read_readvariableop<
8savev2_batch_normalization_312_gamma_read_readvariableop;
7savev2_batch_normalization_312_beta_read_readvariableopB
>savev2_batch_normalization_312_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_312_moving_variance_read_readvariableop0
,savev2_conv2d_442_kernel_read_readvariableop.
*savev2_conv2d_442_bias_read_readvariableop<
8savev2_batch_normalization_313_gamma_read_readvariableop;
7savev2_batch_normalization_313_beta_read_readvariableopB
>savev2_batch_normalization_313_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_313_moving_variance_read_readvariableop.
*savev2_dense_69_kernel_read_readvariableop,
(savev2_dense_69_bias_read_readvariableop<
8savev2_batch_normalization_314_gamma_read_readvariableop;
7savev2_batch_normalization_314_beta_read_readvariableopB
>savev2_batch_normalization_314_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_314_moving_variance_read_readvariableop3
/savev2_gender_output_kernel_read_readvariableop1
-savev2_gender_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_440_kernel_m_read_readvariableop5
1savev2_adam_conv2d_440_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_311_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_311_beta_m_read_readvariableop7
3savev2_adam_conv2d_441_kernel_m_read_readvariableop5
1savev2_adam_conv2d_441_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_312_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_312_beta_m_read_readvariableop7
3savev2_adam_conv2d_442_kernel_m_read_readvariableop5
1savev2_adam_conv2d_442_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_313_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_313_beta_m_read_readvariableop5
1savev2_adam_dense_69_kernel_m_read_readvariableop3
/savev2_adam_dense_69_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_314_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_314_beta_m_read_readvariableop:
6savev2_adam_gender_output_kernel_m_read_readvariableop8
4savev2_adam_gender_output_bias_m_read_readvariableop7
3savev2_adam_conv2d_440_kernel_v_read_readvariableop5
1savev2_adam_conv2d_440_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_311_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_311_beta_v_read_readvariableop7
3savev2_adam_conv2d_441_kernel_v_read_readvariableop5
1savev2_adam_conv2d_441_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_312_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_312_beta_v_read_readvariableop7
3savev2_adam_conv2d_442_kernel_v_read_readvariableop5
1savev2_adam_conv2d_442_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_313_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_313_beta_v_read_readvariableop5
1savev2_adam_dense_69_kernel_v_read_readvariableop3
/savev2_adam_dense_69_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_314_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_314_beta_v_read_readvariableop:
6savev2_adam_gender_output_kernel_v_read_readvariableop8
4savev2_adam_gender_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ½'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*æ&
valueÜ&BÙ&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*¥
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B  
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_440_kernel_read_readvariableop*savev2_conv2d_440_bias_read_readvariableop8savev2_batch_normalization_311_gamma_read_readvariableop7savev2_batch_normalization_311_beta_read_readvariableop>savev2_batch_normalization_311_moving_mean_read_readvariableopBsavev2_batch_normalization_311_moving_variance_read_readvariableop,savev2_conv2d_441_kernel_read_readvariableop*savev2_conv2d_441_bias_read_readvariableop8savev2_batch_normalization_312_gamma_read_readvariableop7savev2_batch_normalization_312_beta_read_readvariableop>savev2_batch_normalization_312_moving_mean_read_readvariableopBsavev2_batch_normalization_312_moving_variance_read_readvariableop,savev2_conv2d_442_kernel_read_readvariableop*savev2_conv2d_442_bias_read_readvariableop8savev2_batch_normalization_313_gamma_read_readvariableop7savev2_batch_normalization_313_beta_read_readvariableop>savev2_batch_normalization_313_moving_mean_read_readvariableopBsavev2_batch_normalization_313_moving_variance_read_readvariableop*savev2_dense_69_kernel_read_readvariableop(savev2_dense_69_bias_read_readvariableop8savev2_batch_normalization_314_gamma_read_readvariableop7savev2_batch_normalization_314_beta_read_readvariableop>savev2_batch_normalization_314_moving_mean_read_readvariableopBsavev2_batch_normalization_314_moving_variance_read_readvariableop/savev2_gender_output_kernel_read_readvariableop-savev2_gender_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_440_kernel_m_read_readvariableop1savev2_adam_conv2d_440_bias_m_read_readvariableop?savev2_adam_batch_normalization_311_gamma_m_read_readvariableop>savev2_adam_batch_normalization_311_beta_m_read_readvariableop3savev2_adam_conv2d_441_kernel_m_read_readvariableop1savev2_adam_conv2d_441_bias_m_read_readvariableop?savev2_adam_batch_normalization_312_gamma_m_read_readvariableop>savev2_adam_batch_normalization_312_beta_m_read_readvariableop3savev2_adam_conv2d_442_kernel_m_read_readvariableop1savev2_adam_conv2d_442_bias_m_read_readvariableop?savev2_adam_batch_normalization_313_gamma_m_read_readvariableop>savev2_adam_batch_normalization_313_beta_m_read_readvariableop1savev2_adam_dense_69_kernel_m_read_readvariableop/savev2_adam_dense_69_bias_m_read_readvariableop?savev2_adam_batch_normalization_314_gamma_m_read_readvariableop>savev2_adam_batch_normalization_314_beta_m_read_readvariableop6savev2_adam_gender_output_kernel_m_read_readvariableop4savev2_adam_gender_output_bias_m_read_readvariableop3savev2_adam_conv2d_440_kernel_v_read_readvariableop1savev2_adam_conv2d_440_bias_v_read_readvariableop?savev2_adam_batch_normalization_311_gamma_v_read_readvariableop>savev2_adam_batch_normalization_311_beta_v_read_readvariableop3savev2_adam_conv2d_441_kernel_v_read_readvariableop1savev2_adam_conv2d_441_bias_v_read_readvariableop?savev2_adam_batch_normalization_312_gamma_v_read_readvariableop>savev2_adam_batch_normalization_312_beta_v_read_readvariableop3savev2_adam_conv2d_442_kernel_v_read_readvariableop1savev2_adam_conv2d_442_bias_v_read_readvariableop?savev2_adam_batch_normalization_313_gamma_v_read_readvariableop>savev2_adam_batch_normalization_313_beta_v_read_readvariableop1savev2_adam_dense_69_kernel_v_read_readvariableop/savev2_adam_dense_69_bias_v_read_readvariableop?savev2_adam_batch_normalization_314_gamma_v_read_readvariableop>savev2_adam_batch_normalization_314_beta_v_read_readvariableop6savev2_adam_gender_output_kernel_v_read_readvariableop4savev2_adam_gender_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*·
_input_shapes¥
¢: ::::::: : : : : : : @:@:@:@:@:@:
::::::	:: : : : : : : : : ::::: : : : : @:@:@:@:
::::	:::::: : : : : @:@:@:@:
::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :,$(
&
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:&0"
 
_output_shapes
:
:!1

_output_shapes	
::!2

_output_shapes	
::!3

_output_shapes	
::%4!

_output_shapes
:	: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
: : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
: @: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:&B"
 
_output_shapes
:
:!C

_output_shapes	
::!D

_output_shapes	
::!E

_output_shapes	
::%F!

_output_shapes
:	: G

_output_shapes
::H

_output_shapes
: 
÷I
â
E__inference_model_53_layer_call_and_return_conditional_losses_1229662
input_56,
conv2d_440_1229595: 
conv2d_440_1229597:-
batch_normalization_311_1229600:-
batch_normalization_311_1229602:-
batch_normalization_311_1229604:-
batch_normalization_311_1229606:,
conv2d_441_1229610:  
conv2d_441_1229612: -
batch_normalization_312_1229615: -
batch_normalization_312_1229617: -
batch_normalization_312_1229619: -
batch_normalization_312_1229621: ,
conv2d_442_1229625: @ 
conv2d_442_1229627:@-
batch_normalization_313_1229630:@-
batch_normalization_313_1229632:@-
batch_normalization_313_1229634:@-
batch_normalization_313_1229636:@$
dense_69_1229641:

dense_69_1229643:	.
batch_normalization_314_1229646:	.
batch_normalization_314_1229648:	.
batch_normalization_314_1229650:	.
batch_normalization_314_1229652:	(
gender_output_1229656:	#
gender_output_1229658:
identity¢/batch_normalization_311/StatefulPartitionedCall¢/batch_normalization_312/StatefulPartitionedCall¢/batch_normalization_313/StatefulPartitionedCall¢/batch_normalization_314/StatefulPartitionedCall¢"conv2d_440/StatefulPartitionedCall¢"conv2d_441/StatefulPartitionedCall¢"conv2d_442/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢%gender_output/StatefulPartitionedCall
"conv2d_440/StatefulPartitionedCallStatefulPartitionedCallinput_56conv2d_440_1229595conv2d_440_1229597*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1229088¥
/batch_normalization_311/StatefulPartitionedCallStatefulPartitionedCall+conv2d_440/StatefulPartitionedCall:output:0batch_normalization_311_1229600batch_normalization_311_1229602batch_normalization_311_1229604batch_normalization_311_1229606*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228782
!max_pooling2d_114/PartitionedCallPartitionedCall8batch_normalization_311/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1228833ª
"conv2d_441/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_114/PartitionedCall:output:0conv2d_441_1229610conv2d_441_1229612*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1229115¥
/batch_normalization_312/StatefulPartitionedCallStatefulPartitionedCall+conv2d_441/StatefulPartitionedCall:output:0batch_normalization_312_1229615batch_normalization_312_1229617batch_normalization_312_1229619batch_normalization_312_1229621*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228858
!max_pooling2d_115/PartitionedCallPartitionedCall8batch_normalization_312/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1228909ª
"conv2d_442/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_115/PartitionedCall:output:0conv2d_442_1229625conv2d_442_1229627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1229142¥
/batch_normalization_313/StatefulPartitionedCallStatefulPartitionedCall+conv2d_442/StatefulPartitionedCall:output:0batch_normalization_313_1229630batch_normalization_313_1229632batch_normalization_313_1229634batch_normalization_313_1229636*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228934
!max_pooling2d_116/PartitionedCallPartitionedCall8batch_normalization_313/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1228985å
flatten_52/PartitionedCallPartitionedCall*max_pooling2d_116/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_52_layer_call_and_return_conditional_losses_1229164
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_69_1229641dense_69_1229643*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_1229177
/batch_normalization_314/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0batch_normalization_314_1229646batch_normalization_314_1229648batch_normalization_314_1229650batch_normalization_314_1229652*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229012ó
dropout_72/PartitionedCallPartitionedCall8batch_normalization_314/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229197§
%gender_output/StatefulPartitionedCallStatefulPartitionedCall#dropout_72/PartitionedCall:output:0gender_output_1229656gender_output_1229658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gender_output_layer_call_and_return_conditional_losses_1229210}
IdentityIdentity.gender_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp0^batch_normalization_311/StatefulPartitionedCall0^batch_normalization_312/StatefulPartitionedCall0^batch_normalization_313/StatefulPartitionedCall0^batch_normalization_314/StatefulPartitionedCall#^conv2d_440/StatefulPartitionedCall#^conv2d_441/StatefulPartitionedCall#^conv2d_442/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall&^gender_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_311/StatefulPartitionedCall/batch_normalization_311/StatefulPartitionedCall2b
/batch_normalization_312/StatefulPartitionedCall/batch_normalization_312/StatefulPartitionedCall2b
/batch_normalization_313/StatefulPartitionedCall/batch_normalization_313/StatefulPartitionedCall2b
/batch_normalization_314/StatefulPartitionedCall/batch_normalization_314/StatefulPartitionedCall2H
"conv2d_440/StatefulPartitionedCall"conv2d_440/StatefulPartitionedCall2H
"conv2d_441/StatefulPartitionedCall"conv2d_441/StatefulPartitionedCall2H
"conv2d_442/StatefulPartitionedCall"conv2d_442/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2N
%gender_output/StatefulPartitionedCall%gender_output/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
input_56
É
c
G__inference_flatten_52_layer_call_and_return_conditional_losses_1229164

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Á
O
3__inference_max_pooling2d_115_layer_call_fn_1230317

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1228909
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_311_layer_call_fn_1230184

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228813
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
·
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229012

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
H
,__inference_flatten_52_layer_call_fn_1230419

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_52_layer_call_and_return_conditional_losses_1229164a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å
®
*__inference_model_53_layer_call_fn_1229854

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_53_layer_call_and_return_conditional_losses_1229217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1228985

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_conv2d_442_layer_call_and_return_conditional_losses_1230342

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_313_layer_call_fn_1230355

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1228934
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
e
G__inference_dropout_72_layer_call_and_return_conditional_losses_1229197

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²%
ñ
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230525

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1228833

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
Ø
9__inference_batch_normalization_314_layer_call_fn_1230471

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229059p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ø
9__inference_batch_normalization_314_layer_call_fn_1230458

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229012p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ã
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230312

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_312_layer_call_fn_1230263

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1228858
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó
µ
"__inference__wrapped_model_1228760
input_56L
2model_53_conv2d_440_conv2d_readvariableop_resource:A
3model_53_conv2d_440_biasadd_readvariableop_resource:F
8model_53_batch_normalization_311_readvariableop_resource:H
:model_53_batch_normalization_311_readvariableop_1_resource:W
Imodel_53_batch_normalization_311_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_53_batch_normalization_311_fusedbatchnormv3_readvariableop_1_resource:L
2model_53_conv2d_441_conv2d_readvariableop_resource: A
3model_53_conv2d_441_biasadd_readvariableop_resource: F
8model_53_batch_normalization_312_readvariableop_resource: H
:model_53_batch_normalization_312_readvariableop_1_resource: W
Imodel_53_batch_normalization_312_fusedbatchnormv3_readvariableop_resource: Y
Kmodel_53_batch_normalization_312_fusedbatchnormv3_readvariableop_1_resource: L
2model_53_conv2d_442_conv2d_readvariableop_resource: @A
3model_53_conv2d_442_biasadd_readvariableop_resource:@F
8model_53_batch_normalization_313_readvariableop_resource:@H
:model_53_batch_normalization_313_readvariableop_1_resource:@W
Imodel_53_batch_normalization_313_fusedbatchnormv3_readvariableop_resource:@Y
Kmodel_53_batch_normalization_313_fusedbatchnormv3_readvariableop_1_resource:@D
0model_53_dense_69_matmul_readvariableop_resource:
@
1model_53_dense_69_biasadd_readvariableop_resource:	Q
Bmodel_53_batch_normalization_314_batchnorm_readvariableop_resource:	U
Fmodel_53_batch_normalization_314_batchnorm_mul_readvariableop_resource:	S
Dmodel_53_batch_normalization_314_batchnorm_readvariableop_1_resource:	S
Dmodel_53_batch_normalization_314_batchnorm_readvariableop_2_resource:	H
5model_53_gender_output_matmul_readvariableop_resource:	D
6model_53_gender_output_biasadd_readvariableop_resource:
identity¢@model_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp¢Bmodel_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1¢/model_53/batch_normalization_311/ReadVariableOp¢1model_53/batch_normalization_311/ReadVariableOp_1¢@model_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp¢Bmodel_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1¢/model_53/batch_normalization_312/ReadVariableOp¢1model_53/batch_normalization_312/ReadVariableOp_1¢@model_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp¢Bmodel_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1¢/model_53/batch_normalization_313/ReadVariableOp¢1model_53/batch_normalization_313/ReadVariableOp_1¢9model_53/batch_normalization_314/batchnorm/ReadVariableOp¢;model_53/batch_normalization_314/batchnorm/ReadVariableOp_1¢;model_53/batch_normalization_314/batchnorm/ReadVariableOp_2¢=model_53/batch_normalization_314/batchnorm/mul/ReadVariableOp¢*model_53/conv2d_440/BiasAdd/ReadVariableOp¢)model_53/conv2d_440/Conv2D/ReadVariableOp¢*model_53/conv2d_441/BiasAdd/ReadVariableOp¢)model_53/conv2d_441/Conv2D/ReadVariableOp¢*model_53/conv2d_442/BiasAdd/ReadVariableOp¢)model_53/conv2d_442/Conv2D/ReadVariableOp¢(model_53/dense_69/BiasAdd/ReadVariableOp¢'model_53/dense_69/MatMul/ReadVariableOp¢-model_53/gender_output/BiasAdd/ReadVariableOp¢,model_53/gender_output/MatMul/ReadVariableOp¤
)model_53/conv2d_440/Conv2D/ReadVariableOpReadVariableOp2model_53_conv2d_440_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ã
model_53/conv2d_440/Conv2DConv2Dinput_561model_53/conv2d_440/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

*model_53/conv2d_440/BiasAdd/ReadVariableOpReadVariableOp3model_53_conv2d_440_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
model_53/conv2d_440/BiasAddBiasAdd#model_53/conv2d_440/Conv2D:output:02model_53/conv2d_440/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
model_53/conv2d_440/ReluRelu$model_53/conv2d_440/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¤
/model_53/batch_normalization_311/ReadVariableOpReadVariableOp8model_53_batch_normalization_311_readvariableop_resource*
_output_shapes
:*
dtype0¨
1model_53/batch_normalization_311/ReadVariableOp_1ReadVariableOp:model_53_batch_normalization_311_readvariableop_1_resource*
_output_shapes
:*
dtype0Æ
@model_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_53_batch_normalization_311_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ê
Bmodel_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_53_batch_normalization_311_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0û
1model_53/batch_normalization_311/FusedBatchNormV3FusedBatchNormV3&model_53/conv2d_440/Relu:activations:07model_53/batch_normalization_311/ReadVariableOp:value:09model_53/batch_normalization_311/ReadVariableOp_1:value:0Hmodel_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( Ñ
"model_53/max_pooling2d_114/MaxPoolMaxPool5model_53/batch_normalization_311/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
)model_53/conv2d_441/Conv2D/ReadVariableOpReadVariableOp2model_53_conv2d_441_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0æ
model_53/conv2d_441/Conv2DConv2D+model_53/max_pooling2d_114/MaxPool:output:01model_53/conv2d_441/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

*model_53/conv2d_441/BiasAdd/ReadVariableOpReadVariableOp3model_53_conv2d_441_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¹
model_53/conv2d_441/BiasAddBiasAdd#model_53/conv2d_441/Conv2D:output:02model_53/conv2d_441/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model_53/conv2d_441/ReluRelu$model_53/conv2d_441/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
/model_53/batch_normalization_312/ReadVariableOpReadVariableOp8model_53_batch_normalization_312_readvariableop_resource*
_output_shapes
: *
dtype0¨
1model_53/batch_normalization_312/ReadVariableOp_1ReadVariableOp:model_53_batch_normalization_312_readvariableop_1_resource*
_output_shapes
: *
dtype0Æ
@model_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_53_batch_normalization_312_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ê
Bmodel_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_53_batch_normalization_312_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0û
1model_53/batch_normalization_312/FusedBatchNormV3FusedBatchNormV3&model_53/conv2d_441/Relu:activations:07model_53/batch_normalization_312/ReadVariableOp:value:09model_53/batch_normalization_312/ReadVariableOp_1:value:0Hmodel_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( Ñ
"model_53/max_pooling2d_115/MaxPoolMaxPool5model_53/batch_normalization_312/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
¤
)model_53/conv2d_442/Conv2D/ReadVariableOpReadVariableOp2model_53_conv2d_442_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0æ
model_53/conv2d_442/Conv2DConv2D+model_53/max_pooling2d_115/MaxPool:output:01model_53/conv2d_442/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

*model_53/conv2d_442/BiasAdd/ReadVariableOpReadVariableOp3model_53_conv2d_442_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¹
model_53/conv2d_442/BiasAddBiasAdd#model_53/conv2d_442/Conv2D:output:02model_53/conv2d_442/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_53/conv2d_442/ReluRelu$model_53/conv2d_442/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
/model_53/batch_normalization_313/ReadVariableOpReadVariableOp8model_53_batch_normalization_313_readvariableop_resource*
_output_shapes
:@*
dtype0¨
1model_53/batch_normalization_313/ReadVariableOp_1ReadVariableOp:model_53_batch_normalization_313_readvariableop_1_resource*
_output_shapes
:@*
dtype0Æ
@model_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_53_batch_normalization_313_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ê
Bmodel_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_53_batch_normalization_313_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0û
1model_53/batch_normalization_313/FusedBatchNormV3FusedBatchNormV3&model_53/conv2d_442/Relu:activations:07model_53/batch_normalization_313/ReadVariableOp:value:09model_53/batch_normalization_313/ReadVariableOp_1:value:0Hmodel_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( Ñ
"model_53/max_pooling2d_116/MaxPoolMaxPool5model_53/batch_normalization_313/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
j
model_53/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ª
model_53/flatten_52/ReshapeReshape+model_53/max_pooling2d_116/MaxPool:output:0"model_53/flatten_52/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_53/dense_69/MatMul/ReadVariableOpReadVariableOp0model_53_dense_69_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
model_53/dense_69/MatMulMatMul$model_53/flatten_52/Reshape:output:0/model_53/dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model_53/dense_69/BiasAdd/ReadVariableOpReadVariableOp1model_53_dense_69_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
model_53/dense_69/BiasAddBiasAdd"model_53/dense_69/MatMul:product:00model_53/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
model_53/dense_69/ReluRelu"model_53/dense_69/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
9model_53/batch_normalization_314/batchnorm/ReadVariableOpReadVariableOpBmodel_53_batch_normalization_314_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0u
0model_53/batch_normalization_314/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Û
.model_53/batch_normalization_314/batchnorm/addAddV2Amodel_53/batch_normalization_314/batchnorm/ReadVariableOp:value:09model_53/batch_normalization_314/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0model_53/batch_normalization_314/batchnorm/RsqrtRsqrt2model_53/batch_normalization_314/batchnorm/add:z:0*
T0*
_output_shapes	
:Á
=model_53/batch_normalization_314/batchnorm/mul/ReadVariableOpReadVariableOpFmodel_53_batch_normalization_314_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
.model_53/batch_normalization_314/batchnorm/mulMul4model_53/batch_normalization_314/batchnorm/Rsqrt:y:0Emodel_53/batch_normalization_314/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ä
0model_53/batch_normalization_314/batchnorm/mul_1Mul$model_53/dense_69/Relu:activations:02model_53/batch_normalization_314/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
;model_53/batch_normalization_314/batchnorm/ReadVariableOp_1ReadVariableOpDmodel_53_batch_normalization_314_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ö
0model_53/batch_normalization_314/batchnorm/mul_2MulCmodel_53/batch_normalization_314/batchnorm/ReadVariableOp_1:value:02model_53/batch_normalization_314/batchnorm/mul:z:0*
T0*
_output_shapes	
:½
;model_53/batch_normalization_314/batchnorm/ReadVariableOp_2ReadVariableOpDmodel_53_batch_normalization_314_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0Ö
.model_53/batch_normalization_314/batchnorm/subSubCmodel_53/batch_normalization_314/batchnorm/ReadVariableOp_2:value:04model_53/batch_normalization_314/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ö
0model_53/batch_normalization_314/batchnorm/add_1AddV24model_53/batch_normalization_314/batchnorm/mul_1:z:02model_53/batch_normalization_314/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_53/dropout_72/IdentityIdentity4model_53/batch_normalization_314/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,model_53/gender_output/MatMul/ReadVariableOpReadVariableOp5model_53_gender_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¶
model_53/gender_output/MatMulMatMul%model_53/dropout_72/Identity:output:04model_53/gender_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-model_53/gender_output/BiasAdd/ReadVariableOpReadVariableOp6model_53_gender_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
model_53/gender_output/BiasAddBiasAdd'model_53/gender_output/MatMul:product:05model_53/gender_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_53/gender_output/SoftmaxSoftmax'model_53/gender_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(model_53/gender_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
NoOpNoOpA^model_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOpC^model_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp_10^model_53/batch_normalization_311/ReadVariableOp2^model_53/batch_normalization_311/ReadVariableOp_1A^model_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOpC^model_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp_10^model_53/batch_normalization_312/ReadVariableOp2^model_53/batch_normalization_312/ReadVariableOp_1A^model_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOpC^model_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp_10^model_53/batch_normalization_313/ReadVariableOp2^model_53/batch_normalization_313/ReadVariableOp_1:^model_53/batch_normalization_314/batchnorm/ReadVariableOp<^model_53/batch_normalization_314/batchnorm/ReadVariableOp_1<^model_53/batch_normalization_314/batchnorm/ReadVariableOp_2>^model_53/batch_normalization_314/batchnorm/mul/ReadVariableOp+^model_53/conv2d_440/BiasAdd/ReadVariableOp*^model_53/conv2d_440/Conv2D/ReadVariableOp+^model_53/conv2d_441/BiasAdd/ReadVariableOp*^model_53/conv2d_441/Conv2D/ReadVariableOp+^model_53/conv2d_442/BiasAdd/ReadVariableOp*^model_53/conv2d_442/Conv2D/ReadVariableOp)^model_53/dense_69/BiasAdd/ReadVariableOp(^model_53/dense_69/MatMul/ReadVariableOp.^model_53/gender_output/BiasAdd/ReadVariableOp-^model_53/gender_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : 2
@model_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp@model_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp2
Bmodel_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp_1Bmodel_53/batch_normalization_311/FusedBatchNormV3/ReadVariableOp_12b
/model_53/batch_normalization_311/ReadVariableOp/model_53/batch_normalization_311/ReadVariableOp2f
1model_53/batch_normalization_311/ReadVariableOp_11model_53/batch_normalization_311/ReadVariableOp_12
@model_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp@model_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp2
Bmodel_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp_1Bmodel_53/batch_normalization_312/FusedBatchNormV3/ReadVariableOp_12b
/model_53/batch_normalization_312/ReadVariableOp/model_53/batch_normalization_312/ReadVariableOp2f
1model_53/batch_normalization_312/ReadVariableOp_11model_53/batch_normalization_312/ReadVariableOp_12
@model_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp@model_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp2
Bmodel_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp_1Bmodel_53/batch_normalization_313/FusedBatchNormV3/ReadVariableOp_12b
/model_53/batch_normalization_313/ReadVariableOp/model_53/batch_normalization_313/ReadVariableOp2f
1model_53/batch_normalization_313/ReadVariableOp_11model_53/batch_normalization_313/ReadVariableOp_12v
9model_53/batch_normalization_314/batchnorm/ReadVariableOp9model_53/batch_normalization_314/batchnorm/ReadVariableOp2z
;model_53/batch_normalization_314/batchnorm/ReadVariableOp_1;model_53/batch_normalization_314/batchnorm/ReadVariableOp_12z
;model_53/batch_normalization_314/batchnorm/ReadVariableOp_2;model_53/batch_normalization_314/batchnorm/ReadVariableOp_22~
=model_53/batch_normalization_314/batchnorm/mul/ReadVariableOp=model_53/batch_normalization_314/batchnorm/mul/ReadVariableOp2X
*model_53/conv2d_440/BiasAdd/ReadVariableOp*model_53/conv2d_440/BiasAdd/ReadVariableOp2V
)model_53/conv2d_440/Conv2D/ReadVariableOp)model_53/conv2d_440/Conv2D/ReadVariableOp2X
*model_53/conv2d_441/BiasAdd/ReadVariableOp*model_53/conv2d_441/BiasAdd/ReadVariableOp2V
)model_53/conv2d_441/Conv2D/ReadVariableOp)model_53/conv2d_441/Conv2D/ReadVariableOp2X
*model_53/conv2d_442/BiasAdd/ReadVariableOp*model_53/conv2d_442/BiasAdd/ReadVariableOp2V
)model_53/conv2d_442/Conv2D/ReadVariableOp)model_53/conv2d_442/Conv2D/ReadVariableOp2T
(model_53/dense_69/BiasAdd/ReadVariableOp(model_53/dense_69/BiasAdd/ReadVariableOp2R
'model_53/dense_69/MatMul/ReadVariableOp'model_53/dense_69/MatMul/ReadVariableOp2^
-model_53/gender_output/BiasAdd/ReadVariableOp-model_53/gender_output/BiasAdd/ReadVariableOp2\
,model_53/gender_output/MatMul/ReadVariableOp,model_53/gender_output/MatMul/ReadVariableOp:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
input_56

j
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1230322

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
O
3__inference_max_pooling2d_114_layer_call_fn_1230225

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1228833
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²%
ñ
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1229059

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_311_layer_call_fn_1230171

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1228782
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
c
G__inference_flatten_52_layer_call_and_return_conditional_losses_1230425

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
E
input_569
serving_default_input_56:0ÿÿÿÿÿÿÿÿÿ  A
gender_output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op"
_tf_keras_layer
ê
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance"
_tf_keras_layer
¥
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op"
_tf_keras_layer
ê
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance"
_tf_keras_layer
¥
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op"
_tf_keras_layer
ê
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\axis
	]gamma
^beta
_moving_mean
`moving_variance"
_tf_keras_layer
¥
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
»
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias"
_tf_keras_layer
ê
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{axis
	|gamma
}beta
~moving_mean
moving_variance"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
è
0
 1
)2
*3
+4
,5
96
:7
C8
D9
E10
F11
S12
T13
]14
^15
_16
`17
s18
t19
|20
}21
~22
23
24
25"
trackable_list_wrapper
¨
0
 1
)2
*3
94
:5
C6
D7
S8
T9
]10
^11
s12
t13
|14
}15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
å
trace_0
trace_1
trace_2
trace_32ò
*__inference_model_53_layer_call_fn_1229272
*__inference_model_53_layer_call_fn_1229854
*__inference_model_53_layer_call_fn_1229911
*__inference_model_53_layer_call_fn_1229592¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ñ
trace_0
trace_1
trace_2
trace_32Þ
E__inference_model_53_layer_call_and_return_conditional_losses_1230014
E__inference_model_53_layer_call_and_return_conditional_losses_1230138
E__inference_model_53_layer_call_and_return_conditional_losses_1229662
E__inference_model_53_layer_call_and_return_conditional_losses_1229732¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
ÎBË
"__inference__wrapped_model_1228760input_56"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ä
	iter
beta_1
beta_2

decay
 learning_ratem m)m*m9m:mCmDm Sm¡Tm¢]m£^m¤sm¥tm¦|m§}m¨	m©	mªv« v¬)v­*v®9v¯:v°Cv±Dv²Sv³Tv´]vµ^v¶sv·tv¸|v¹}vº	v»	v¼"
	optimizer
-
¡serving_default"
signature_map
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò
§trace_02Ó
,__inference_conv2d_440_layer_call_fn_1230147¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0

¨trace_02î
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1230158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¨trace_0
+:)2conv2d_440/kernel
:2conv2d_440/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ç
®trace_0
¯trace_12¬
9__inference_batch_normalization_311_layer_call_fn_1230171
9__inference_batch_normalization_311_layer_call_fn_1230184³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0z¯trace_1

°trace_0
±trace_12â
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230202
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230220³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z°trace_0z±trace_1
 "
trackable_list_wrapper
+:)2batch_normalization_311/gamma
*:(2batch_normalization_311/beta
3:1 (2#batch_normalization_311/moving_mean
7:5 (2'batch_normalization_311/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ù
·trace_02Ú
3__inference_max_pooling2d_114_layer_call_fn_1230225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0

¸trace_02õ
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1230230¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¸trace_0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
ò
¾trace_02Ó
,__inference_conv2d_441_layer_call_fn_1230239¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0

¿trace_02î
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1230250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¿trace_0
+:) 2conv2d_441/kernel
: 2conv2d_441/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
C0
D1
E2
F3"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
ç
Åtrace_0
Ætrace_12¬
9__inference_batch_normalization_312_layer_call_fn_1230263
9__inference_batch_normalization_312_layer_call_fn_1230276³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0zÆtrace_1

Çtrace_0
Ètrace_12â
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230294
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230312³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÇtrace_0zÈtrace_1
 "
trackable_list_wrapper
+:) 2batch_normalization_312/gamma
*:( 2batch_normalization_312/beta
3:1  (2#batch_normalization_312/moving_mean
7:5  (2'batch_normalization_312/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
ù
Îtrace_02Ú
3__inference_max_pooling2d_115_layer_call_fn_1230317¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÎtrace_0

Ïtrace_02õ
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1230322¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÏtrace_0
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ò
Õtrace_02Ó
,__inference_conv2d_442_layer_call_fn_1230331¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÕtrace_0

Ötrace_02î
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1230342¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÖtrace_0
+:) @2conv2d_442/kernel
:@2conv2d_442/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
ç
Ütrace_0
Ýtrace_12¬
9__inference_batch_normalization_313_layer_call_fn_1230355
9__inference_batch_normalization_313_layer_call_fn_1230368³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÜtrace_0zÝtrace_1

Þtrace_0
ßtrace_12â
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230386
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230404³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÞtrace_0zßtrace_1
 "
trackable_list_wrapper
+:)@2batch_normalization_313/gamma
*:(@2batch_normalization_313/beta
3:1@ (2#batch_normalization_313/moving_mean
7:5@ (2'batch_normalization_313/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
ù
åtrace_02Ú
3__inference_max_pooling2d_116_layer_call_fn_1230409¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zåtrace_0

ætrace_02õ
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1230414¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zætrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
ò
ìtrace_02Ó
,__inference_flatten_52_layer_call_fn_1230419¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zìtrace_0

ítrace_02î
G__inference_flatten_52_layer_call_and_return_conditional_losses_1230425¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zítrace_0
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
ð
ótrace_02Ñ
*__inference_dense_69_layer_call_fn_1230434¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zótrace_0

ôtrace_02ì
E__inference_dense_69_layer_call_and_return_conditional_losses_1230445¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zôtrace_0
#:!
2dense_69/kernel
:2dense_69/bias
<
|0
}1
~2
3"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
ç
útrace_0
ûtrace_12¬
9__inference_batch_normalization_314_layer_call_fn_1230458
9__inference_batch_normalization_314_layer_call_fn_1230471³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zútrace_0zûtrace_1

ütrace_0
ýtrace_12â
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230491
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230525³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zütrace_0zýtrace_1
 "
trackable_list_wrapper
,:*2batch_normalization_314/gamma
+:)2batch_normalization_314/beta
4:2 (2#batch_normalization_314/moving_mean
8:6 (2'batch_normalization_314/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Í
trace_0
trace_12
,__inference_dropout_72_layer_call_fn_1230530
,__inference_dropout_72_layer_call_fn_1230535³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12È
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230540
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230552³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
õ
trace_02Ö
/__inference_gender_output_layer_call_fn_1230561¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ñ
J__inference_gender_output_layer_call_and_return_conditional_losses_1230572¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
':%	2gender_output/kernel
 :2gender_output/bias
X
+0
,1
E2
F3
_4
`5
~6
7"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
*__inference_model_53_layer_call_fn_1229272input_56"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_model_53_layer_call_fn_1229854inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_model_53_layer_call_fn_1229911inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
*__inference_model_53_layer_call_fn_1229592input_56"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_53_layer_call_and_return_conditional_losses_1230014inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_53_layer_call_and_return_conditional_losses_1230138inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_53_layer_call_and_return_conditional_losses_1229662input_56"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_53_layer_call_and_return_conditional_losses_1229732input_56"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÍBÊ
%__inference_signature_wrapper_1229797input_56"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_conv2d_440_layer_call_fn_1230147inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1230158inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
9__inference_batch_normalization_311_layer_call_fn_1230171inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
9__inference_batch_normalization_311_layer_call_fn_1230184inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230202inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230220inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
çBä
3__inference_max_pooling2d_114_layer_call_fn_1230225inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1230230inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_conv2d_441_layer_call_fn_1230239inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1230250inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
9__inference_batch_normalization_312_layer_call_fn_1230263inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
9__inference_batch_normalization_312_layer_call_fn_1230276inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230294inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230312inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
çBä
3__inference_max_pooling2d_115_layer_call_fn_1230317inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1230322inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_conv2d_442_layer_call_fn_1230331inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1230342inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
9__inference_batch_normalization_313_layer_call_fn_1230355inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
9__inference_batch_normalization_313_layer_call_fn_1230368inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230386inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230404inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
çBä
3__inference_max_pooling2d_116_layer_call_fn_1230409inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1230414inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_flatten_52_layer_call_fn_1230419inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_flatten_52_layer_call_and_return_conditional_losses_1230425inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_69_layer_call_fn_1230434inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_69_layer_call_and_return_conditional_losses_1230445inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
9__inference_batch_normalization_314_layer_call_fn_1230458inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
9__inference_batch_normalization_314_layer_call_fn_1230471inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230491inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230525inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
,__inference_dropout_72_layer_call_fn_1230530inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
,__inference_dropout_72_layer_call_fn_1230535inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230540inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230552inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ãBà
/__inference_gender_output_layer_call_fn_1230561inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
J__inference_gender_output_layer_call_and_return_conditional_losses_1230572inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0:.2Adam/conv2d_440/kernel/m
": 2Adam/conv2d_440/bias/m
0:.2$Adam/batch_normalization_311/gamma/m
/:-2#Adam/batch_normalization_311/beta/m
0:. 2Adam/conv2d_441/kernel/m
":  2Adam/conv2d_441/bias/m
0:. 2$Adam/batch_normalization_312/gamma/m
/:- 2#Adam/batch_normalization_312/beta/m
0:. @2Adam/conv2d_442/kernel/m
": @2Adam/conv2d_442/bias/m
0:.@2$Adam/batch_normalization_313/gamma/m
/:-@2#Adam/batch_normalization_313/beta/m
(:&
2Adam/dense_69/kernel/m
!:2Adam/dense_69/bias/m
1:/2$Adam/batch_normalization_314/gamma/m
0:.2#Adam/batch_normalization_314/beta/m
,:*	2Adam/gender_output/kernel/m
%:#2Adam/gender_output/bias/m
0:.2Adam/conv2d_440/kernel/v
": 2Adam/conv2d_440/bias/v
0:.2$Adam/batch_normalization_311/gamma/v
/:-2#Adam/batch_normalization_311/beta/v
0:. 2Adam/conv2d_441/kernel/v
":  2Adam/conv2d_441/bias/v
0:. 2$Adam/batch_normalization_312/gamma/v
/:- 2#Adam/batch_normalization_312/beta/v
0:. @2Adam/conv2d_442/kernel/v
": @2Adam/conv2d_442/bias/v
0:.@2$Adam/batch_normalization_313/gamma/v
/:-@2#Adam/batch_normalization_313/beta/v
(:&
2Adam/dense_69/kernel/v
!:2Adam/dense_69/bias/v
1:/2$Adam/batch_normalization_314/gamma/v
0:.2#Adam/batch_normalization_314/beta/v
,:*	2Adam/gender_output/kernel/v
%:#2Adam/gender_output/bias/v¿
"__inference__wrapped_model_1228760 )*+,9:CDEFST]^_`st|~}9¢6
/¢,
*'
input_56ÿÿÿÿÿÿÿÿÿ  
ª "=ª:
8
gender_output'$
gender_outputÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230202)*+,M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_311_layer_call_and_return_conditional_losses_1230220)*+,M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_311_layer_call_fn_1230171)*+,M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_311_layer_call_fn_1230184)*+,M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230294CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ï
T__inference_batch_normalization_312_layer_call_and_return_conditional_losses_1230312CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ç
9__inference_batch_normalization_312_layer_call_fn_1230263CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ç
9__inference_batch_normalization_312_layer_call_fn_1230276CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ï
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230386]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ï
T__inference_batch_normalization_313_layer_call_and_return_conditional_losses_1230404]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ç
9__inference_batch_normalization_313_layer_call_fn_1230355]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ç
9__inference_batch_normalization_313_layer_call_fn_1230368]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¼
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230491d|~}4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¼
T__inference_batch_normalization_314_layer_call_and_return_conditional_losses_1230525d~|}4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_314_layer_call_fn_1230458W|~}4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_314_layer_call_fn_1230471W~|}4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
G__inference_conv2d_440_layer_call_and_return_conditional_losses_1230158l 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_440_layer_call_fn_1230147_ 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_441_layer_call_and_return_conditional_losses_1230250l9:7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_441_layer_call_fn_1230239_9:7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ ·
G__inference_conv2d_442_layer_call_and_return_conditional_losses_1230342lST7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_442_layer_call_fn_1230331_ST7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@§
E__inference_dense_69_layer_call_and_return_conditional_losses_1230445^st0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_69_layer_call_fn_1230434Qst0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230540^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ©
G__inference_dropout_72_layer_call_and_return_conditional_losses_1230552^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dropout_72_layer_call_fn_1230530Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dropout_72_layer_call_fn_1230535Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¬
G__inference_flatten_52_layer_call_and_return_conditional_losses_1230425a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_flatten_52_layer_call_fn_1230419T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ­
J__inference_gender_output_layer_call_and_return_conditional_losses_1230572_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_gender_output_layer_call_fn_1230561R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿñ
N__inference_max_pooling2d_114_layer_call_and_return_conditional_losses_1230230R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_114_layer_call_fn_1230225R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
N__inference_max_pooling2d_115_layer_call_and_return_conditional_losses_1230322R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_115_layer_call_fn_1230317R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
N__inference_max_pooling2d_116_layer_call_and_return_conditional_losses_1230414R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_116_layer_call_fn_1230409R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
E__inference_model_53_layer_call_and_return_conditional_losses_1229662 )*+,9:CDEFST]^_`st|~}A¢>
7¢4
*'
input_56ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_model_53_layer_call_and_return_conditional_losses_1229732 )*+,9:CDEFST]^_`st~|}A¢>
7¢4
*'
input_56ÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ð
E__inference_model_53_layer_call_and_return_conditional_losses_1230014 )*+,9:CDEFST]^_`st|~}?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ð
E__inference_model_53_layer_call_and_return_conditional_losses_1230138 )*+,9:CDEFST]^_`st~|}?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ©
*__inference_model_53_layer_call_fn_1229272{ )*+,9:CDEFST]^_`st|~}A¢>
7¢4
*'
input_56ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
*__inference_model_53_layer_call_fn_1229592{ )*+,9:CDEFST]^_`st~|}A¢>
7¢4
*'
input_56ÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ§
*__inference_model_53_layer_call_fn_1229854y )*+,9:CDEFST]^_`st|~}?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ§
*__inference_model_53_layer_call_fn_1229911y )*+,9:CDEFST]^_`st~|}?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿÎ
%__inference_signature_wrapper_1229797¤ )*+,9:CDEFST]^_`st|~}E¢B
¢ 
;ª8
6
input_56*'
input_56ÿÿÿÿÿÿÿÿÿ  "=ª:
8
gender_output'$
gender_outputÿÿÿÿÿÿÿÿÿ